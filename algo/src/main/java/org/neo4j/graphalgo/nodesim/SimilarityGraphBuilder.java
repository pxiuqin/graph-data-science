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
package org.neo4j.graphalgo.nodesim;

import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.huge.TransientAdjacencyList;
import org.neo4j.graphalgo.core.huge.TransientAdjacencyOffsets;
import org.neo4j.graphalgo.core.huge.HugeGraph;
import org.neo4j.graphalgo.core.loading.HugeGraphUtil;
import org.neo4j.graphalgo.core.loading.IdMap;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;

import java.util.concurrent.ExecutorService;
import java.util.stream.Stream;

class SimilarityGraphBuilder {

    private final ExecutorService executorService;
    private final AllocationTracker tracker;

    static MemoryEstimation memoryEstimation(int topK, int topN) {
        return MemoryEstimations.setup("", (dimensions, concurrency) -> {
            long maxNodesToCompare = Math.min(dimensions.maxRelCount(), dimensions.nodeCount());
            long maxNumberOfSimilarityResults = maxNodesToCompare * (maxNodesToCompare - 1) / 2;

            long newNodeCount = maxNodesToCompare;
            long newRelationshipCount = maxNumberOfSimilarityResults;

            if (topN > 0) {
                newRelationshipCount = Math.min(newRelationshipCount, topN);
                // If we reduce the number of relationships via topN,
                // we also have a new upper bound of the number of
                // nodes connected by those relationships.
                // The upper bound is a graph consisting of disjoint node pairs.
                newNodeCount = Math.min(maxNodesToCompare, newRelationshipCount * 2);
            }

            int averageDegree = Math.toIntExact(newRelationshipCount / newNodeCount);
            // For topK, we duplicate each similarity pair, which leads to a higher average degree.
            // At the same time, we limit the average degree by topK.
            if (topK > 0) {
                averageDegree = Math.min(Math.toIntExact(2 * newRelationshipCount / newNodeCount), topK);
            }

            return MemoryEstimations.builder(HugeGraph.class)
                .add(
                    "adjacency list",
                    TransientAdjacencyList.compressedMemoryEstimation(averageDegree, newNodeCount)
                )
                .add("adjacency offsets", TransientAdjacencyOffsets.memoryEstimation(concurrency, newNodeCount))
                .build();
        });
    }

    private final Graph baseGraph;

    private final IdMap baseIdMap;

    SimilarityGraphBuilder(
        Graph baseGraph,
        ExecutorService executorService,
        AllocationTracker tracker
    ) {
        this.executorService = executorService;
        this.tracker = tracker;
        this.baseGraph = baseGraph;

        if (baseGraph.nodeMapping() instanceof IdMap) {
            this.baseIdMap = (IdMap) baseGraph.nodeMapping();
        } else {
            throw new IllegalArgumentException("Base graph must contain an IdMap.");
        }
    }

    Graph build(Stream<SimilarityResult> stream) {
        Orientation orientation = baseGraph.isUndirected() ? Orientation.UNDIRECTED : Orientation.NATURAL;
        HugeGraphUtil.RelationshipsBuilder relationshipsBuilder = HugeGraphUtil.createRelImporter(
            baseIdMap,
            orientation,
            true,
            Aggregation.NONE,
            executorService,
            tracker
        );
        relationshipsBuilder.addFromInternal(stream);
        return HugeGraphUtil.create(
            baseIdMap,
            relationshipsBuilder.build(),
            tracker
        );
    }
}
