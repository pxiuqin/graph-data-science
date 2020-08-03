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

//构建含权重的PageRank的计算单元
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
        for (long nodeId = startNode; nodeId < endNode; ++nodeId) {
            delta = deltas[(int) (nodeId - startNode)];  //取出delta
            if (delta > 0.0) {
                int degree = degrees.degree(nodeId);
                if (degree > 0) {
                    sumOfWeights = aggregatedDegrees.get(nodeId); //权重和
                    rels.forEachRelationship(nodeId, DEFAULT_WEIGHT, this);
                }
            }
            progressLogger.logProgress(graph.degree(nodeId));
        }
    }

    @Override
    public boolean accept(long sourceNodeId, long targetNodeId, double property) {
        if (property > 0) {
            double proportion = property / sumOfWeights;
            float srcRankDelta = (float) (delta * proportion);  //乘以权重带过来的传播因子
            if (srcRankDelta != 0F) {
                int idx = binaryLookup(targetNodeId, starts);  //找到索引位置
                nextScores[idx][(int) (targetNodeId - starts[idx])] += srcRankDelta;  //赋值出链携带pr值
            }
        }

        return true;
    }
}
