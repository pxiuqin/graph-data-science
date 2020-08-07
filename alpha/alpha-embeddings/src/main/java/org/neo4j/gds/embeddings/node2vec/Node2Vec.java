/*
 * Copyright (c) 2017-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.neo4j.gds.embeddings.node2vec;

import org.apache.commons.lang3.mutable.MutableLong;
import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;

/**
 * node2vec的思想同DeepWalk一样：生成随机游走，对随机游走采样得到（节点，上下文）的组合，
 * 然后用处理词向量的方法对这样的组合建模得到网络节点的表示。本方法针对生成随机游走过程中做了一些创新，给定q和p参数来控制随机游走情况
 *
 * 算法逻辑【参考论文：node2vec: Scalable Feature Learning for Networks】：
 * 算法的参数，图G、表示向量维度d、每个节点生成的游走个数r，游走长度l，上下文的窗口长度k，以及之前提到的p、q参数。
 *
 * 1、根据p、q和之前的公式计算一个节点到它的邻居的转移概率。
 * 2、将这个转移概率加到图G中形成G'。
 * 3、walks用来存储随机游走，先初始化为空。
 * 4、外循环r次表示每个节点作为初始节点要生成r个随机游走。
 * 5、然后对图中每个节点。
 * 6、生成一条随机游走walk。
 * 7、将walk添加到walks中保存。
 * 8、然后用SGD的方法对walks进行训练。
 *
 * 第6步中一条walk的生成方式如下：
 *
 * 1、将初始节点u添加进去。
 * 2、walk的长度为l，因此还要再循环添加l-1个节点。
 * 3、当前节点设为walk最后添加的节点。
 * 4、找出当前节点的所有邻居节点。
 * 5、根据转移概率采样选择某个邻居s。【这里采用AliasSample，别名采样，应用场景:加权采样,即按照随机事件出现的概率抽样】
 * 6、将该邻居添加到walk中。
 */
public class Node2Vec extends Algorithm<Node2Vec, HugeObjectArray<Vector>> {

    private final Graph graph;
    private final Node2VecBaseConfig config;
    private final AllocationTracker tracker;

    public Node2Vec(Graph graph, Node2VecBaseConfig config, ProgressLogger progressLogger, AllocationTracker tracker) {
        this.graph = graph;
        this.config = config;
        this.progressLogger = progressLogger;
        this.tracker = tracker;
    }

    @Override
    public HugeObjectArray<Vector> compute() {
        RandomWalk randomWalk = new RandomWalk(
            graph,
            config.walkLength(),  //随机游走的的步数
            new RandomWalk.NextNodeStrategy(graph, config.returnFactor(), config.inOutFactor()),  //构建一个找下一个节点的策略
            config.concurrency(),
            config.walksPerNode(),
            config.walkBufferSize()  //队列大小
        );

        HugeObjectArray<long[]> walks = HugeObjectArray.newArray(
            long[].class,
            graph.nodeCount() * config.walksPerNode(),
            tracker
        );  //记录下节点随机游走的结果
        MutableLong counter = new MutableLong(0);
        randomWalk
            .compute()
            .forEach(walk -> {
                walks.set(counter.longValue(), walk);  //给定计数添加
                counter.increment();
            });

        var probabilityComputer = new ProbabilityComputer(
            walks,
            graph.nodeCount(),
            config.centerSamplingFactor(),
            config.contextSamplingExponent(),
            config.concurrency(),
            tracker
        );

        var node2VecModel = new Node2VecModel(
            graph.nodeCount(),
            config,
            walks,
            probabilityComputer,
            progressLogger
        );

        node2VecModel.train();

        return node2VecModel.getEmbeddings();
    }

    @Override
    public Node2Vec me() {
        return this;
    }

    @Override
    public void release() {

    }
}
