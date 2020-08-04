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

import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.BatchingProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.MemoryUsage;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.logging.Log;

public class PageRankFactory<CONFIG extends PageRankBaseConfig> implements AlgorithmFactory<PageRank, CONFIG> {

    @Override
    public PageRank build(
        Graph graph,
        PageRankBaseConfig configuration,
        AllocationTracker tracker,
        Log log
    ) {
        var progressLogger = new BatchingProgressLogger(
            log,
            graph.relationshipCount(),
            getClass().getSimpleName(),
            configuration.concurrency()
        );

        return algorithmType(configuration).create(
            graph,
            configuration.sourceNodeIds(),
            configuration,
            Pools.DEFAULT,
            progressLogger,
            tracker
        );
    }

    @Override
    public MemoryEstimation memoryEstimation(CONFIG config) {
        return MemoryEstimations.builder(PageRank.class)
            .add(MemoryEstimations.setup("computeSteps", (dimensions, concurrency) -> MemoryEstimations
                .builder(PageRank.ComputeSteps.class)
                .perThread("scores[] wrapper", MemoryUsage::sizeOfObjectArray)
                .perThread("starts[]", MemoryUsage::sizeOfLongArray)
                .perThread("lengths[]", MemoryUsage::sizeOfLongArray)
                .perThread("list of computeSteps", MemoryUsage::sizeOfObjectArray)
                .perThread("ComputeStep", algorithmType(config).memoryEstimation())
                .build()))
            .build();
    }

    private PageRankAlgorithmType algorithmType(PageRankBaseConfig configuration) {
        return configuration.relationshipWeightProperty() == null
            ? PageRankAlgorithmType.NON_WEIGHTED
            : PageRankAlgorithmType.WEIGHTED;
    }
}
