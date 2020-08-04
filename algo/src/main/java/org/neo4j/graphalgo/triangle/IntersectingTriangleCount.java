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
package org.neo4j.graphalgo.triangle;

import org.jetbrains.annotations.TestOnly;
import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.IntersectionConsumer;
import org.neo4j.graphalgo.api.RelationshipIntersect;
import org.neo4j.graphalgo.api.nodeproperties.LongNodeProperties;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicLongArray;

import java.util.Collection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * TriangleCount counts the number of triangles in the Graph as well
 * as the number of triangles that passes through a node.
 *
 * This impl uses another approach where all the triangles can be calculated
 * using set intersection methods of the graph itself.
 *
 * https://epubs.siam.org/doi/pdf/10.1137/1.9781611973198.1
 * http://www.cse.cuhk.edu.hk/~jcheng/papers/triangle_kdd11.pdf
 * https://i11www.iti.kit.edu/extra/publications/sw-fclt-05_t.pdf
 * http://www.math.cmu.edu/~ctsourak/tsourICDM08.pdf
 */
@SuppressWarnings("FieldCanBeLocal")
public class IntersectingTriangleCount extends Algorithm<IntersectingTriangleCount, IntersectingTriangleCount.TriangleCountResult> {

    static final int EXCLUDED_NODE_TRIANGLE_COUNT = -1;

    private Graph graph;
    private final TriangleCountBaseConfig config;
    private ExecutorService executorService;
    private final AtomicLong queue;

    // results
    private final HugeAtomicLongArray triangleCounts;
    private long globalTriangleCount;

    private LongAdder globalTriangleCounter;

    public IntersectingTriangleCount(
        Graph graph,
        TriangleCountBaseConfig config,
        ExecutorService executorService,
        AllocationTracker tracker,
        ProgressLogger progressLogger
    ) {
        this.graph = graph;
        this.config = config;
        this.executorService = executorService;
        triangleCounts = HugeAtomicLongArray.newArray(graph.nodeCount(), tracker);
        globalTriangleCounter = new LongAdder();
        queue = new AtomicLong();
        this.progressLogger = progressLogger;
    }

    @TestOnly
    public IntersectingTriangleCount(
        Graph graph,
        TriangleCountBaseConfig config,
        ExecutorService executorService,
        AllocationTracker tracker
    ) {
        this(graph, config, executorService, tracker, ProgressLogger.NULL_LOGGER);
    }

    @Override
    public final IntersectingTriangleCount me() {
        return this;
    }

    @Override
    public void release() {
        executorService = null;
        graph = null;
        globalTriangleCounter = null;
    }

    @Override
    public TriangleCountResult compute() {
        queue.set(0);
        globalTriangleCounter.reset();
        // create tasks
        final Collection<? extends Runnable> tasks = ParallelUtil.tasks(config.concurrency(), () -> new IntersectTask(graph));
        // run
        ParallelUtil.run(tasks, executorService);

        globalTriangleCount = globalTriangleCounter.longValue();

        return TriangleCountResult.of(
            triangleCounts,
            globalTriangleCount
        );
    }

    private class IntersectTask implements Runnable, IntersectionConsumer {

        private final RelationshipIntersect intersect;

        IntersectTask(Graph graph) {
            intersect = graph.intersection(config.maxDegree());
        }

        @Override
        public void run() {
            long node;
            while ((node = queue.getAndIncrement()) < graph.nodeCount() && running()) {
                if (graph.degree(node) <= config.maxDegree()) {
                    intersect.intersectAll(node, this);
                } else {
                    triangleCounts.set(node, EXCLUDED_NODE_TRIANGLE_COUNT);
                }
                getProgressLogger().logProgress();
            }
        }

        @Override
        public void accept(final long nodeA, final long nodeB, final long nodeC) {
            // only use this triangle where the id's are in order, not the other 5
            if (nodeA < nodeB) { //  && nodeB < nodeC
                triangleCounts.update(nodeA, (previous) -> previous + 1);
                triangleCounts.update(nodeB, (previous) -> previous + 1);
                triangleCounts.update(nodeC, (previous) -> previous + 1);
                globalTriangleCounter.increment();
            }
        }
    }

    @ValueClass
    public interface TriangleCountResult {
        // value at index `i` is number of triangles for node with id `i`
        HugeAtomicLongArray localTriangles();

        long globalTriangles();

        static TriangleCountResult of(
            HugeAtomicLongArray triangles,
            long globalTriangles
        ) {
            return ImmutableTriangleCountResult
                .builder()
                .localTriangles(triangles)
                .globalTriangles(globalTriangles)
                .build();
        }

        default LongNodeProperties asNodeProperties() {
            return localTriangles().asNodeProperties();
        }
    }
}
