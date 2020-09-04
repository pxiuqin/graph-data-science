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
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.Assessable;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;

import java.util.concurrent.ExecutorService;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.core.utils.BitUtil.ceilDiv;

public interface PageRankAlgorithm extends Assessable {

    /**
     * Forces sequential use. If you want parallelism, prefer
     *
     * {@link #create(Graph, LongStream, PageRankBaseConfig, ExecutorService, int, ProgressLogger, AllocationTracker)} }
     */
    default PageRank create(
        Graph graph,
        PageRankBaseConfig algoConfig,
        LongStream sourceNodeIds,
        ProgressLogger progressLogger
    ) {
        return create(graph, sourceNodeIds, algoConfig, null, ParallelUtil.DEFAULT_BATCH_SIZE, progressLogger,
            AllocationTracker.empty()
        );
    }

    default PageRank create(
        Graph graph,
        LongStream sourceNodeIds,
        PageRankBaseConfig algoConfig,
        ExecutorService executor,
        ProgressLogger progressLogger,
        AllocationTracker tracker
    ) {
        return create(
            graph,
            sourceNodeIds,
            algoConfig,
            executor,
            ParallelUtil.DEFAULT_BATCH_SIZE,
            progressLogger,
            tracker
        );
    }

    default PageRank create(
        Graph graph,
        LongStream sourceNodeIds,
        PageRankBaseConfig algoConfig,
        ExecutorService executor,
        int batchSize,
        ProgressLogger progressLogger,
        AllocationTracker tracker
    ) {
        return new PageRank(
            graph,
            variant(algoConfig),
            sourceNodeIds,
            algoConfig,
            executor,
            batchSize,
            progressLogger,
            tracker
        );
    }

    PageRankVariant variant(PageRankBaseConfig config); //基于配置来构建PageRank的各种变体

    Class<? extends BaseComputeStep> computeStepClass();

    @Override
    default MemoryEstimation memoryEstimation() {
        return MemoryEstimations.setup("ComputeStep", (dimensions, concurrency) -> {
            long nodeCount = dimensions.nodeCount();
            long nodesPerThread = ceilDiv(nodeCount, concurrency);
            return BaseComputeStep.estimateMemory((int) nodesPerThread, computeStepClass());
        });
    }
}
