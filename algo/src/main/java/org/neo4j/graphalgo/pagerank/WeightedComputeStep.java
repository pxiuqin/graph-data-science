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
package org.neo4j.graphalgo.pagerank;

import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.api.RelationshipWithPropertyConsumer;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;

import static org.neo4j.graphalgo.core.utils.ArrayUtil.binaryLookup;
import static org.neo4j.graphalgo.pagerank.PageRank.DEFAULT_WEIGHT;

//构建含权重的PageRank的计算单元【这里还实现了边操作的回调操作】
public class WeightedComputeStep extends BaseComputeStep implements RelationshipWithPropertyConsumer {

    private final HugeDoubleArray aggregatedDegrees;
    private double sumOfWeights;
    private double delta;

    WeightedComputeStep(
            double dampingFactor,
            long[] sourceNodeIds,
            Graph graph,
            AllocationTracker tracker,
            int partitionSize,
            long startNode,
            DegreeCache degreeCache,
            ProgressLogger progressLogger
    ) {
        super(dampingFactor,
                sourceNodeIds,
                graph,
                tracker,
                partitionSize,
                startNode,
                progressLogger
        );
        this.aggregatedDegrees = degreeCache.aggregatedDegrees();
    }

    //计算
    void singleIteration() {
        long startNode = this.startNode;
        long endNode = this.endNode;
        RelationshipIterator rels = this.relationshipIterator;

        //从分片中的开始节点到结束节点
        for (long nodeId = startNode; nodeId < endNode; ++nodeId) {
            delta = deltas[(int) (nodeId - startNode)];  //取出delta
            if (delta > 0.0) {
                int degree = degrees.degree(nodeId);  //出度
                if (degree > 0) {
                    //这里的值每次不去保留？因为下面accept中会用到此变量【实现了边操作回调函数】
                    sumOfWeights = aggregatedDegrees.get(nodeId); //权重和代替delta/degree
                    rels.forEachRelationship(nodeId, DEFAULT_WEIGHT, this);  //这里把当前节点的传播权重值保存
                }
            }
            progressLogger.logProgress(graph.degree(nodeId));
        }
    }

    //处理边操作时回回调这里的函数实现具体的操作【上面的rels.forEachRelationship(nodeId, DEFAULT_WEIGHT, this)中标示用this的回调】
    @Override
    public boolean accept(long sourceNodeId, long targetNodeId, double property) {
        if (property > 0) {
            double proportion = property / sumOfWeights;
            float srcRankDelta = (float) (delta * proportion);  //乘以权重带过来的传播因子，这里delta是 delta = deltas[(int) (nodeId - startNode)];处计算的结果
            if (srcRankDelta != 0F) {
                //注意这里会有多个线程下完成从多个入链汇集pr值
                int idx = binaryLookup(targetNodeId, starts);  //找到索引位置【在开始节点数组中找位置，也可以理解为分片的起始位置】

                //targetNodeId - starts[idx]表示targetNodeId的位置
                nextScores[idx][(int) (targetNodeId - starts[idx])] += srcRankDelta;  //赋值携带pr值【这里会有多线程从多个入链来的pr值，所以用了累加】
            }
        }

        return true;
    }
}
