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
package org.neo4j.graphalgo.betweenness;

import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicDoubleArray;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.core.utils.paged.HugeIntArray;
import org.neo4j.graphalgo.core.utils.paged.HugeLongArray;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;
import org.neo4j.logging.Log;

import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfLongArray;

public class BetweennessCentralityFactory<CONFIG extends BetweennessCentralityBaseConfig> implements AlgorithmFactory<BetweennessCentrality, CONFIG> {

    @Override
    public BetweennessCentrality build(Graph graph, CONFIG configuration, AllocationTracker tracker, Log log) {
        var samplingSize = configuration.samplingSize();
        var samplingSeed = configuration.samplingSeed();

        var strategy = samplingSize.isPresent() && samplingSize.get() < graph.nodeCount()
            ? new SelectionStrategy.RandomDegree(samplingSize.get(), samplingSeed)
            : SelectionStrategy.ALL;

        return new BetweennessCentrality(
            graph,
            strategy,
            Pools.DEFAULT,
            configuration.concurrency(),
            tracker
        );
    }

    @Override
    public MemoryEstimation memoryEstimation(CONFIG configuration) {
        return MemoryEstimations.builder(BetweennessCentrality.class)
            .perNode("centrality scores", HugeAtomicDoubleArray::memoryEstimation)
            .perThread("compute task", MemoryEstimations.builder(BetweennessCentrality.BCTask.class)
                .add("predecessors", MemoryEstimations.setup("", (dimensions, concurrency) -> {
                    // Predecessors are represented by LongArrayList which wrap a long[]
                    long averagePredecessorSize = sizeOfLongArray(dimensions.averageDegree());
                    return MemoryEstimations.builder(HugeObjectArray.class)
                        .perNode("array", nodeCount -> nodeCount * averagePredecessorSize)
                        .build();
                }))
                .perNode("forwardNodes", HugeLongArray::memoryEstimation)
                .perNode("backwardNodes", HugeLongArray::memoryEstimation)
                .perNode("deltas", HugeDoubleArray::memoryEstimation)
                .perNode("sigmas", HugeLongArray::memoryEstimation)
                .perNode("distances", HugeIntArray::memoryEstimation)
                .build())
            .build();
    }
}
