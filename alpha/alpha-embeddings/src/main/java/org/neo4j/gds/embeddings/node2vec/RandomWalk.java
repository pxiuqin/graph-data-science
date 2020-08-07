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

import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.RelationshipConsumer;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.queue.QueueBasedSpliterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

//基于随机游走完成节点向量化表示【node2vec算法也就与这个来随机游走，通过q和p参数控制游走策略】
public class RandomWalk extends Algorithm<RandomWalk, Stream<long[]>> {

    private final Graph graph;
    private final int steps;
    private final NextNodeStrategy strategy;
    private final int concurrency;
    private final int walksPerNode;
    private final int queueSize;

    public RandomWalk(
        Graph graph,
        int steps,
        NextNodeStrategy strategy,
        int concurrency,
        int walksPerNode,
        int queueSize
    ) {
        this.graph = graph;
        this.steps = steps;  //随机游走的步数
        this.strategy = strategy;
        this.concurrency = concurrency;
        this.walksPerNode = walksPerNode;
        this.queueSize = queueSize;  //游走节点缓存的队列大小
    }

    @Override
    public Stream<long[]> compute() {
        int minBatchSize = 100;
        int timeout = 100;
        BlockingQueue<long[]> walks = new ArrayBlockingQueue<>(queueSize);
        long[] TOMB = new long[0];

        long batchSize = ParallelUtil.adjustedBatchSize(graph.nodeCount(), concurrency, minBatchSize);  //给定并发线程数，重新分配每个线程的处理节点数
        ArrayList<Runnable> tasks = new ArrayList<>();
        for (var i = 0; i < graph.nodeCount(); i += batchSize) {
            var start = i;
            var stop = Math.min(start + batchSize, graph.nodeCount());  //不能超出节点大小
            tasks.add(
                () -> {
                    //每个任务分配一定节点数量
                    for (var j = start; j < stop; j++) {
                        doWalk(j).forEach(walk -> put(walks, walk));
                    }
                }
            );
        }
        new Thread(() -> {
            ParallelUtil.runWithConcurrency(concurrency, tasks, terminationFlag, Pools.DEFAULT);
            put(walks, TOMB);  //添加一个可以分割的符号
        }).start();
        QueueBasedSpliterator<long[]> spliterator = new QueueBasedSpliterator<>(walks, TOMB, terminationFlag, timeout);
        return StreamSupport.stream(spliterator, false);  //基于可分割的流处理返回数据
    }

    @Override
    public RandomWalk me() {
        return this;
    }

    @Override
    public void release() { }

    private Stream<long[]> doWalk(long startNodeId) {
        //针对每个起始节点迭代指定的次数
        return IntStream.range(0, walksPerNode).mapToObj(ignored -> {
            long[] nodeIds = new long[steps + 1];
            long currentNodeId = startNodeId;
            long previousNodeId = currentNodeId;  //给定当前节点
            nodeIds[0] = currentNodeId;

            //游走的步数
            for (int i = 1; i <= steps; i++) {
                long nextNodeId = strategy.getNextNode(currentNodeId, previousNodeId);  //找到游走的下一个节点
                previousNodeId = currentNodeId;
                currentNodeId = nextNodeId;

                if (currentNodeId == -1 || !terminationFlag.running()) {
                    return Arrays.copyOf(nodeIds, i);   //无下一个节点时返回
                }
                nodeIds[i] = currentNodeId;  //保存当前节点
            }

            return nodeIds;
        });
    }

    private long toOriginalNodeId(long currentNodeId) {
        return currentNodeId == -1 ? -1 : graph.toOriginalNodeId(currentNodeId);
    }

    //结果队列放每个节点游走的序列
    private static <T> void put(BlockingQueue<T> queue, T items) {
        try {
            queue.put(items);
        } catch (InterruptedException e) {}
    }

    //下一个节点遍历策略【通过q和p参数控制策略】
    public static class NextNodeStrategy {
        private final Graph graph;
        private final double returnParam;
        private final double inOutParam;

        //基于P参数：returnParam,Q参数：inOutParam控制游走策略
        public NextNodeStrategy(Graph graph, double returnParam, double inOutParam) {
            this.graph = graph;
            this.returnParam = returnParam;
            this.inOutParam = inOutParam;
        }

        //这里基于node2vec算法中，给定当前节点v和前一个节点t
        public long getNextNode(long currentNode, long previousNode) {
            Graph threadLocalGraph = graph.concurrentCopy();

            int degree = threadLocalGraph.degree(currentNode);
            if (degree == 0) {
                return -1;  //孤立点不关注
            }

            //获取当前节点和邻居节点的转移概率值
            double[] distribution = buildProbabilityDistribution(
                threadLocalGraph,
                currentNode,
                previousNode,
                returnParam,
                inOutParam,
                degree
            );

            //使用AliasSample【别名采样】，应用场景:加权采样,即按照随机事件出现的概率抽样
            int neighbourIndex = pickIndexFromDistribution(distribution, ThreadLocalRandom.current().nextDouble());

            return threadLocalGraph.getTarget(currentNode, neighbourIndex);  //指定当前节点第几个邻居
        }

        //构建概率分布，其实构建了一个节点到他邻居的转移概率
        private double[] buildProbabilityDistribution(
            Graph threadLocalGraph,
            long currentNodeId,
            long previousNodeId,
            double returnParam,
            double inOutParam,
            int degree
        ) {
            ProbabilityDistributionComputer consumer = new ProbabilityDistributionComputer(
                threadLocalGraph.concurrentCopy(),
                degree,
                currentNodeId,
                previousNodeId,
                returnParam,
                inOutParam
            );
            threadLocalGraph.forEachRelationship(currentNodeId, consumer);
            return consumer.probabilities();  //返回正则化后的概率值
        }

        //归一化处理
        private static double[] normalizeDistribution(double[] array, double sum) {
            for (int i = 0; i < array.length; i++) {
                array[i] /= sum;  //占总分
            }
            return array;
        }

        //选择大于阈值的分布数【其实是从邻居中随机选择一些节点，可以理解为AliasSample：应用场景:加权采样,即按照随机事件出现的概率抽样】
        private static int pickIndexFromDistribution(double[] normalizedDistribution, double randomThreshold) {
            double cumulativeProbability = 0.0;
            for (int i = 0; i < normalizedDistribution.length; i++) {
                cumulativeProbability += normalizedDistribution[i];
                if (randomThreshold <= cumulativeProbability) {
                    return i;  //找到累加概率的位置
                }
            }
            return normalizedDistribution.length - 1;
        }

        //分布计算
        private class ProbabilityDistributionComputer implements RelationshipConsumer {
            private final Graph threadLocalGraph;
            final double[] probabilities;
            private final long currentNodeId;
            private final long previousNodeId;
            private final double returnParam;  //表示参数P值
            private final double inOutParam;  //表示参数Q值
            double probSum;
            int index;

            public ProbabilityDistributionComputer(
                Graph threadLocalGraph,
                int degree,
                long currentNodeId,
                long previousNodeId,
                double returnParam,
                double inOutParam
            ) {
                this.threadLocalGraph = threadLocalGraph;
                this.currentNodeId = currentNodeId;
                this.previousNodeId = previousNodeId;
                this.returnParam = returnParam;
                this.inOutParam = inOutParam;
                probabilities = new double[degree];  //当前节点到邻居的转换概率
                probSum = 0;
                index = 0;
            }

            @Override
            public boolean accept(long start, long end) {
                long neighbourId = start == currentNodeId ? end : start;  //确定邻居节点

                double probability;

                /**
                 * 如果t与x相等，那么采样x的概率为 1/p ；
                 * 如果t与x相连，那么采样x的概率1；
                 * 如果t与x不相连，那么采样x概率为 1/q。
                 *
                 * 说明：
                 * t=previousNodeId
                 * x=neighbourId
                 * p=returnParam
                 * q=inOutParam
                 */
                if (neighbourId == previousNodeId) {
                    // node is previous node
                    probability = 1D / returnParam;  //如果邻居节点是前向节点，那么转移概率为 1/p
                } else if (threadLocalGraph.exists(previousNodeId, neighbourId)) {
                    // node is also adjacent to previous node --> distance to previous node is 1
                    probability = 1D;  //如果previousNodeId和neighbourId是邻居节点，那么转移概率为1
                } else {
                    // node is not adjacent to previous node --> distance to previous node is 2
                    probability = 1D / inOutParam;  //如果节点不是邻居，那么转移概率为1/q
                }
                probabilities[index] = probability;
                probSum += probability;  //累加
                index++;
                return true;
            }

            private double[] probabilities() {
                return normalizeDistribution(probabilities, probSum);
            }
        }
    }
}
