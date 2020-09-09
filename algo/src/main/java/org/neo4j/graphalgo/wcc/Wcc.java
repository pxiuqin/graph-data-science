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
package org.neo4j.graphalgo.wcc;

import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.RelationshipConsumer;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.api.RelationshipWithPropertyConsumer;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.dss.DisjointSetStruct;
import org.neo4j.graphalgo.core.utils.paged.dss.HugeAtomicDisjointSetStruct;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutorService;

import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

/**
 * Parallel Union-Find Algorithm based on the
 * "Wait-free Parallel Algorithms for the Union-Find Problem" paper.
 *
 * @see HugeAtomicDisjointSetStruct
 * @see <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.8354&rep=rep1&type=pdf">the paper</a>
 */
public class Wcc extends Algorithm<Wcc, DisjointSetStruct> {

    private final WccBaseConfig config;
    private final NodeProperties initialComponents;
    private final ExecutorService executor;
    private final AllocationTracker tracker;
    private final long nodeCount;
    private final long batchSize;
    private final int threadSize;

    private Graph graph;

    public static MemoryEstimation memoryEstimation(boolean incremental) {
        return MemoryEstimations
            .builder(Wcc.class)
            .add("dss", HugeAtomicDisjointSetStruct.memoryEstimation(incremental))
            .build();
    }

    public Wcc(
        Graph graph,
        ExecutorService executor,
        int minBatchSize,
        WccBaseConfig config,
        ProgressLogger progressLogger,
        AllocationTracker tracker
    ) {
        this.graph = graph;
        this.config = config;
        this.initialComponents = config.isIncremental()
            ? graph.nodeProperties(config.seedProperty())
            : null;
        this.executor = executor;
        this.tracker = tracker;
        this.nodeCount = graph.nodeCount();
        this.batchSize = ParallelUtil.adjustedBatchSize(
            nodeCount,
            config.concurrency(),
            minBatchSize,
            Integer.MAX_VALUE
        );

        long threadSize = ParallelUtil.threadCount(batchSize, nodeCount);
        if (threadSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(formatWithLocale(
                "Too many nodes (%d) to run union find with the given concurrency (%d) and batchSize (%d)",
                nodeCount,
                config.concurrency(),
                batchSize
            ));
        }
        this.threadSize = (int) threadSize;

        this.progressLogger = progressLogger;
    }

    @Override
    public DisjointSetStruct compute() {
        progressLogger.logMessage(":: Start");

        long nodeCount = graph.nodeCount();

        DisjointSetStruct dss = config.isIncremental()
            ? new HugeAtomicDisjointSetStruct(nodeCount, initialComponents, tracker, config.concurrency())
            : new HugeAtomicDisjointSetStruct(nodeCount, tracker, config.concurrency());

        final Collection<Runnable> tasks = new ArrayList<>(threadSize);
        for (long i = 0L; i < this.nodeCount; i += batchSize) {
            WCCTask wccTask = Double.isNaN(threshold()) || threshold() == 0
                ? new WCCTask(dss, i)   //可以分批并行计算
                : new WCCWithThresholdTask(threshold(), dss, i);
            tasks.add(wccTask);
        }
        ParallelUtil.run(tasks, executor);

        progressLogger.logMessage(":: Finished");
        return dss;
    }

    @Override
    public Wcc me() {
        return this;
    }

    @Override
    public void release() {
        graph = null;
    }

    public double threshold() {
        return config.threshold();
    }

    private static double defaultWeight(double threshold) {
        return threshold + 1;
    }

    private class WCCTask implements Runnable, RelationshipConsumer {

        final DisjointSetStruct struct;
        final RelationshipIterator rels;
        private final long offset;
        private final long end;

        WCCTask(DisjointSetStruct struct, long offset) {
            this.struct = struct;
            this.rels = graph.concurrentCopy();
            this.offset = offset;  //分批处理的标识
            this.end = Math.min(offset + batchSize, nodeCount);
        }

        @Override
        public void run() {
            for (long node = offset; node < end; node++) {
                compute(node);   //每个节点处理下
                if (node % RUN_CHECK_NODE_COUNT == 0) {
                    assertRunning();
                }

                getProgressLogger().logProgress(graph.degree(node));
            }
        }

        void compute(final long node) {
            rels.forEachRelationship(node, this);
        }

        @Override
        public boolean accept(final long sourceNodeId, final long targetNodeId) {
            struct.union(sourceNodeId, targetNodeId);  //因为图是连通的所以要合并
            return true;
        }
    }

    private class WCCWithThresholdTask extends WCCTask implements RelationshipWithPropertyConsumer {

        private final double threshold;

        WCCWithThresholdTask(double threshold, DisjointSetStruct struct, long offset) {
            super(struct, offset);
            this.threshold = threshold;
        }

        @Override
        void compute(final long node) {
            rels.forEachRelationship(node, Wcc.defaultWeight(threshold), this);
        }

        @Override
        public boolean accept(final long sourceNodeId, final long targetNodeId, final double property) {
            if (property > threshold) {  //大于阈值
                struct.union(sourceNodeId, targetNodeId);
            }
            return true;
        }
    }
}
