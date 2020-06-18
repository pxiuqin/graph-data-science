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
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;

//含权重的的PageRank算法变体
public class WeightedPageRankVariant implements PageRankVariant {
    private final boolean cacheWeights;

    WeightedPageRankVariant(boolean cacheWeights) {
        this.cacheWeights = cacheWeights;
    }

    @Override
    public ComputeStep createComputeStep(
            double dampingFactor,
            double toleranceValue,
            long[] sourceNodeIds,
            Graph graph,
            AllocationTracker tracker,
            int partitionCount,
            long start,
            DegreeCache aggregatedDegrees,
            long nodeCount,
            ProgressLogger progressLogger
    ) {
        return new WeightedComputeStep(
                dampingFactor,
                sourceNodeIds,
                graph,
                tracker,
                partitionCount,
                start,
                aggregatedDegrees,
                progressLogger
        );
    }

    @Override
    public DegreeComputer degreeComputer(Graph graph) {
        return new WeightedDegreeComputer(graph, cacheWeights);
    }
}
