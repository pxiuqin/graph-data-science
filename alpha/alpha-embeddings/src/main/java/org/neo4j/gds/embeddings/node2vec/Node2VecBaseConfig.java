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

import org.immutables.value.Value;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.config.AlgoBaseConfig;

//Node2Vec算法的配置参数
public interface Node2VecBaseConfig extends AlgoBaseConfig {

    @Value.Default
    @Configuration.IntegerRange(min = 1)
    default int walkLength() {
        return 80;
    }  //随机游走的步数

    @Value.Default
    @Configuration.IntegerRange(min = 2)
    default int walksPerNode() {
        return 10;
    }  //每个节点游走的步数

    @Value.Default
    @Configuration.IntegerRange(min = 2)
    default int windowSize() {
        return 10;
    }   //窗口大小，用用SGD

    @Value.Default
    @Configuration.IntegerRange(min = 1)
    default int walkBufferSize() {
        return 1000;
    }

    /**
     * 如果 q>1 ，那么游走会倾向于在起始点周围的节点之间跑，可以反映出一个节点的BFS特性。
     * 如果 q<1 ，那么游走会倾向于往远处跑，反映出DFS特性。
     * @return
     */
    @Value.Default
    @Configuration.DoubleRange(min = 0.0)
    default double inOutFactor() {
        return 1.0;
    }  //表示出入参数：q

    /**
     * 如果 p>max(q,1) ，那么采样会尽量不往回走，对应上图的情况，就是下一个节点不太可能是上一个访问的节点t【t是出发节点】
     * 如果 p<min(q,1) ，那么采样会更倾向于返回上一个节点，这样就会一直在起始点周围某些节点来回转来转去。
     * @return
     */
    @Value.Default
    @Configuration.DoubleRange(min = 0.0)
    default double returnFactor() {
        return 1.0;
    }  //表示算法中返回概率：p

    @Value.Default
    @Configuration.IntegerRange(min = 1)
    default int negativeSamplingRate() {
        return 5;
    }  //负采样个数

    @Value.Default
    @Configuration.DoubleRange(min = 0.00001, minInclusive = false, max=1.0)
    default double centerSamplingFactor() {
        return 0.001;
    }

    @Value.Default
    @Configuration.DoubleRange(min = 0.00001, minInclusive = false, max=1.0)
    default double contextSamplingExponent() {
        return 0.75;
    }

    @Value.Default
    @Configuration.IntegerRange(min = 1)
    default int embeddingSize() {
        return 128;
    }

    @Value.Default
    @Configuration.DoubleRange(min = 0.0, minInclusive = false)
    default double initialLearningRate() {
        return 0.025;
    }  //学习率

    @Value.Default
    @Configuration.DoubleRange(min = 0.0, minInclusive = false)
    default double minLearningRate() {
        return 0.0001;
    }

    @Value.Default
    default int iterations() {
        return 1;
    }

}
