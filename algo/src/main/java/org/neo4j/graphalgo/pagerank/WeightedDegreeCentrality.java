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

import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.config.ConcurrencyConfig;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

//基于权重的度中心性计算
public class WeightedDegreeCentrality extends Algorithm<WeightedDegreeCentrality, WeightedDegreeCentrality> {
    private final long nodeCount;
    private Graph graph;
    private final ExecutorService executor;
    private final int concurrency;
    private volatile AtomicInteger nodeQueue = new AtomicInteger();
    private final boolean cacheWeights;

    private HugeDoubleArray degrees;
    private HugeObjectArray<HugeDoubleArray> weights;  //用来缓存权重
    private AllocationTracker tracker;

    public WeightedDegreeCentrality(
        Graph graph,
        int concurrency,
        boolean cacheWeights,
        ExecutorService executor,
        AllocationTracker tracker
    ) {
        this.cacheWeights = cacheWeights;
        if (!graph.hasRelationshipProperty()) {
            throw new UnsupportedOperationException("WeightedDegreeCentrality requires a weight property to be loaded.");
        }

        this.tracker = tracker;
        if (concurrency <= 0) {
            concurrency = ConcurrencyConfig.DEFAULT_CONCURRENCY;
        }

        this.graph = graph;
        this.executor = executor;
        this.concurrency = concurrency;
        nodeCount = graph.nodeCount();
        degrees = HugeDoubleArray.newArray(nodeCount, tracker);
        weights = HugeObjectArray.newArray(HugeDoubleArray.class, nodeCount, tracker);
    }

    @Override
    public WeightedDegreeCentrality compute() {
        nodeQueue.set(0);

        long batchSize = ParallelUtil.adjustedBatchSize(nodeCount, concurrency);
        long threadSize = ParallelUtil.threadCount(batchSize, nodeCount);
        if (threadSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(formatWithLocale(
                    "A concurrency of %d is too small to divide graph into at most Integer.MAX_VALUE tasks",
                    concurrency));
        }
        final List<Runnable> tasks = new ArrayList<>((int) threadSize);

        for (int i = 0; i < threadSize; i++) {
            if(cacheWeights) {
                tasks.add(new CacheDegreeTask());
            } else {
                tasks.add(new DegreeTask());
            }
        }
        ParallelUtil.runWithConcurrency(concurrency, tasks, executor);

        return this;
    }

    @Override
    public WeightedDegreeCentrality me() {
        return this;
    }

    @Override
    public void release() {
        graph = null;
    }

    private class DegreeTask implements Runnable {
        @Override
        public void run() {
            final RelationshipIterator threadLocalGraph = graph.concurrentCopy();
            while (true) {
                final int nodeId = nodeQueue.getAndIncrement();
                if (nodeId >= nodeCount || !running()) {
                    return;
                }

                double[] weightedDegree = new double[1];
                
                //迭代计算每条边
                threadLocalGraph.forEachRelationship(nodeId, Double.NaN, (sourceNodeId, targetNodeId, weight) -> {
                    if(weight > 0) {
                        weightedDegree[0] += weight;  //累加权重
                    }

                    return true;
                });

                degrees.set(nodeId, weightedDegree[0]);  //度为其权重

            }
        }
    }

    private class CacheDegreeTask implements Runnable {
        @Override
        public void run() {
            final RelationshipIterator threadLocalGraph = graph.concurrentCopy();
            double[] weightedDegree = new double[1];
            for (; ; ) {
                final int nodeId = nodeQueue.getAndIncrement();
                if (nodeId >= nodeCount || !running()) {
                    return;
                }

                final HugeDoubleArray nodeWeights = HugeDoubleArray.newArray(graph.degree(nodeId), tracker);
                weights.set(nodeId, nodeWeights);  //缓存权重值

                int[] index = {0};
                weightedDegree[0] = 0D;
                threadLocalGraph.forEachRelationship(nodeId, Double.NaN, (sourceNodeId, targetNodeId, weight) -> {
                    if(weight > 0) {
                        weightedDegree[0] += weight;
                    }

                    nodeWeights.set(index[0], weight);
                    index[0]++;
                    return true;
                });

                degrees.set(nodeId, weightedDegree[0]);

            }
        }
    }

    public HugeDoubleArray degrees() {
        return degrees;
    }
    public HugeObjectArray<HugeDoubleArray> weights() {
        return weights;
    }

}
