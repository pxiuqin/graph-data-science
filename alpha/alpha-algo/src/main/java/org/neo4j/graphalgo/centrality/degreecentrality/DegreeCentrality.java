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
package org.neo4j.graphalgo.centrality.degreecentrality;

import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.result.CentralityResult;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

public class DegreeCentrality extends Algorithm<DegreeCentrality, DegreeCentrality> {
    public static final double DEFAULT_WEIGHT = 0D;
    private final int nodeCount;
    private final boolean weighted;
    private Graph graph;
    private final ExecutorService executor;
    private final int concurrency;
    private final HugeDoubleArray result;

    public DegreeCentrality(
            Graph graph,
            ExecutorService executor,
            int concurrency,
            boolean weighted,
            AllocationTracker tracker) {
        this.graph = graph;
        this.executor = executor;
        this.concurrency = concurrency;
        this.nodeCount = Math.toIntExact(graph.nodeCount());
        this.weighted = weighted;
        this.result = HugeDoubleArray.newArray(nodeCount, tracker);
    }

    @Override
    public DegreeCentrality compute() {
        int batchSize = ParallelUtil.adjustedBatchSize(nodeCount, concurrency);
        int taskCount = ParallelUtil.threadCount(batchSize, nodeCount);
        List<Runnable> tasks = new ArrayList<>(taskCount);

        long[] starts = new long[taskCount];
        double[][] partitions = new double[taskCount][batchSize];

        long startNode = 0L;
        for (int i = 0; i < taskCount; i++) {
            starts[i] = startNode;
            if (weighted) {
                tasks.add(new WeightedDegreeTask(graph.concurrentCopy(), starts[i], partitions[i]));
            } else {
                tasks.add(new DegreeTask(graph, starts[i], partitions[i]));
            }
            startNode += batchSize;
        }
        ParallelUtil.runWithConcurrency(concurrency, tasks, executor);

        return this;
    }

    public Algorithm<?, ?> algorithm() {
        return this;
    }

    @Override
    public DegreeCentrality me() {
        return this;
    }

    @Override
    public void release() {
        graph = null;
    }

    public CentralityResult result() {
        return new CentralityResult(result);
    }

    private class DegreeTask implements Runnable {
        private final Graph graph;
        private final long startNodeId;
        private final double[] partition;
        private final long endNodeId;

        DegreeTask(Graph graph, long start, double[] partition) {
            this.graph = graph;
            this.startNodeId = start;
            this.partition = partition;
            this.endNodeId = Math.min(start + partition.length, nodeCount);
        }

        @Override
        public void run() {
            for (long nodeId = startNodeId; nodeId < endNodeId && running(); nodeId++) {
                partition[Math.toIntExact(nodeId - startNodeId)] = graph.degree(nodeId);
            }
            result.copyFromArrayIntoSlice(partition, startNodeId, endNodeId);
        }
    }

    private class WeightedDegreeTask implements Runnable {
        private final RelationshipIterator relationshipIterator;
        private final long startNodeId;
        private final double[] partition;
        private final long endNodeId;

        WeightedDegreeTask(
            RelationshipIterator relationshipIterator,
            long start,
            double[] partition
        ) {
            this.relationshipIterator = relationshipIterator;
            this.startNodeId = start;
            this.partition = partition;
            this.endNodeId = Math.min(start + partition.length, nodeCount);
        }

        @Override
        public void run() {
            for (long nodeId = startNodeId; nodeId < endNodeId && running(); nodeId++) {
                int index = Math.toIntExact(nodeId - startNodeId);
                relationshipIterator.forEachRelationship(nodeId, DEFAULT_WEIGHT, (sourceNodeId, targetNodeId, weight) -> {
                    if (weight > 0) {
                        partition[index] += weight;
                    }
                    return true;
                });
            }

            result.copyFromArrayIntoSlice(partition, startNodeId, endNodeId);
        }
    }
}
