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
package org.neo4j.graphalgo.labelpropagation;

import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.loading.NullPropertyMap.DoubleNullPropertyMap;
import org.neo4j.graphalgo.core.loading.NullPropertyMap.LongNullPropertyMap;
import org.neo4j.graphalgo.core.utils.LazyBatchCollection;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongCollections;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterable;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeLongArray;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;

import static java.util.concurrent.TimeUnit.MICROSECONDS;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;
import static org.neo4j.kernel.api.StatementConstants.NO_SUCH_LABEL;

public class LabelPropagation extends Algorithm<LabelPropagation, LabelPropagation> {

    public static final double DEFAULT_WEIGHT = 1.0;

    private final long nodeCount;
    private final AllocationTracker tracker;
    private final NodeProperties nodeProperties;
    private final NodeProperties nodeWeights;
    private final LabelPropagationBaseConfig config;
    private final ExecutorService executor;

    private Graph graph;
    private HugeLongArray labels;
    private final long maxLabelId;
    private long ranIterations;
    private boolean didConverge;
    private int batchSize;

    public LabelPropagation(
        Graph graph,
        LabelPropagationBaseConfig config,
        ExecutorService executor,
        ProgressLogger progressLogger,
        AllocationTracker tracker
    ) {
        this.graph = graph;
        this.nodeCount = graph.nodeCount();
        this.config = config;
        this.executor = executor;
        this.tracker = tracker;
        this.batchSize = ParallelUtil.DEFAULT_BATCH_SIZE;

        NodeProperties seedProperty;
        String seedPropertyKey = config.seedProperty();
        if (seedPropertyKey != null && graph.availableNodeProperties().contains(seedPropertyKey)) {
            seedProperty = graph.nodeProperties(seedPropertyKey);
        } else {
            seedProperty = new LongNullPropertyMap(0);
        }
        this.nodeProperties = seedProperty;

        NodeProperties nodeWeightProperty;
        String nodeWeightPropertyKey = config.nodeWeightProperty();
        if (nodeWeightPropertyKey != null && graph.availableNodeProperties().contains(nodeWeightPropertyKey)) {
            nodeWeightProperty = graph.nodeProperties(nodeWeightPropertyKey);
        } else {
            nodeWeightProperty = new DoubleNullPropertyMap(1.0);
        }
        this.nodeWeights = nodeWeightProperty;

        maxLabelId = nodeProperties.getMaxPropertyValue().orElse(NO_SUCH_LABEL);

        this.progressLogger = progressLogger;
    }

    @Override
    public LabelPropagation me() {
        return this;
    }

    @Override
    public void release() {
        graph = null;
    }

    public long ranIterations() {
        return ranIterations;
    }

    public boolean didConverge() {
        return didConverge;
    }

    public HugeLongArray labels() {
        return labels;
    }

    @Override
    public LabelPropagation compute() {
        if (config.maxIterations() <= 0L) {
            throw new IllegalArgumentException("Must iterate at least 1 time");
        }

        getProgressLogger().logMessage(":: Start");

        if (labels == null || labels.size() != nodeCount) {
            labels = HugeLongArray.newArray(nodeCount, tracker);
        }

        ranIterations = 0L;
        didConverge = false;

        List<StepRunner> stepRunners = stepRunners();

        while (ranIterations < config.maxIterations()) {
            getProgressLogger().logMessage(formatWithLocale(":: Iteration %d :: Start", ranIterations + 1));
            ParallelUtil.runWithConcurrency(config.concurrency(), stepRunners, 1L, MICROSECONDS, terminationFlag, executor);
            ++ranIterations;
            didConverge = stepRunners.stream().allMatch(StepRunner::didConverge);
            if (didConverge) {
                break;
            }
            getProgressLogger().logMessage(formatWithLocale(":: Iteration %d :: Finished", ranIterations));
            getProgressLogger().reset(graph.relationshipCount());
        }

        stepRunners.forEach(StepRunner::release);
        getProgressLogger().logMessage(":: Finished");

        return me();
    }

    private List<StepRunner> stepRunners() {
        long nodeCount = graph.nodeCount();
        long batchSize = ParallelUtil.adjustedBatchSize(nodeCount, this.batchSize);

        Collection<PrimitiveLongIterable> nodeBatches = LazyBatchCollection.of(
            nodeCount,
            batchSize,
            (start, length) -> () -> PrimitiveLongCollections.range(start, start + length - 1L)
        );

        int threads = nodeBatches.size();
        List<StepRunner> tasks = new ArrayList<>(threads);
        for (PrimitiveLongIterable iter : nodeBatches) {
            InitStep initStep = new InitStep(
                graph,
                nodeProperties,
                nodeWeights,
                iter,
                labels,
                getProgressLogger(),
                maxLabelId
            );
            StepRunner task = new StepRunner(initStep);
            tasks.add(task);
        }
        progressLogger.logMessage(":: Initialization :: Start");
        ParallelUtil.runWithConcurrency(config.concurrency(), tasks, 1, MICROSECONDS, terminationFlag, executor);
        progressLogger.logMessage(":: Initialization :: Finished");
        progressLogger.reset(graph.relationshipCount());
        return tasks;
    }

    void withBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
