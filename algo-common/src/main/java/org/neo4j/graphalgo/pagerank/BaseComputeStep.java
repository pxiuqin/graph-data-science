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

import org.neo4j.graphalgo.api.Degrees;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.MemoryUsage;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;

import java.util.Arrays;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfDoubleArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfFloatArray;

//初步实现计算单元的抽象类，给定了计算的步骤和结果的组装，具体计算需要实现类来完成
public abstract class BaseComputeStep implements ComputeStep {
    /**
     * 计算状态维护
     */
    private static final int S_INIT = 0;
    private static final int S_CALC = 1;
    private static final int S_SYNC = 2;
    private static final int S_NORM = 3;


    private int state;

    long[] starts;
    private int[] lengths;
    protected double tolerance;
    private long[] sourceNodeIds;
    final RelationshipIterator relationshipIterator;
    final Degrees degrees;
    private final AllocationTracker tracker;

    private final double alpha;  //用1-阻尼因子表示
    final double dampingFactor;  //pr中的阻尼因子

    double[] pageRank;
    double[] deltas;
    float[][] nextScores;  //出链分配的分值
    float[][] prevScores;  //入链携带的分值

    final long startNode;
    final ProgressLogger progressLogger;
    final Graph graph;
    final long endNode;
    private final int partitionSize;
    double l2Norm;

    private boolean shouldBreak;

    BaseComputeStep(
        double dampingFactor,  //阻尼因子
        long[] sourceNodeIds,
        Graph graph,
        AllocationTracker tracker,
        int partitionSize,
        long startNode,
        ProgressLogger progressLogger
    ) {
        this(
            dampingFactor,
            PageRank.DEFAULT_TOLERANCE,
            sourceNodeIds,
            graph,
            tracker,
            partitionSize,
            startNode,
            progressLogger
        );
    }

    BaseComputeStep(
        double dampingFactor,
        double tolerance,
        long[] sourceNodeIds,
        Graph graph,
        AllocationTracker tracker,
        int partitionSize,
        long startNode,
        ProgressLogger progressLogger
    ) {
        this.dampingFactor = dampingFactor;
        this.alpha = 1.0 - dampingFactor;
        this.tolerance = tolerance;
        this.sourceNodeIds = sourceNodeIds;
        this.graph = graph;
        this.relationshipIterator = graph.concurrentCopy();
        this.degrees = graph;
        this.tracker = tracker;
        this.partitionSize = partitionSize;
        this.startNode = startNode;
        this.progressLogger = progressLogger;
        this.endNode = startNode + (long) partitionSize;
        state = S_INIT;
    }

    static MemoryEstimation estimateMemory(
        final int partitionSize,
        final Class<?> computeStep
    ) {
        return MemoryEstimations.builder(computeStep)
            .perThread("nextScores[] wrapper", MemoryUsage::sizeOfObjectArray)
            .perThread("inner nextScores[][]", sizeOfFloatArray(partitionSize))
            .fixed("pageRank[]", sizeOfDoubleArray(partitionSize))
            .fixed("deltas[]", sizeOfDoubleArray(partitionSize))
            .build();
    }

    public void setStarts(long[] starts, int[] lengths) {
        this.starts = starts;
        this.lengths = lengths;
    }

    @Override
    public void run() {
        //计算状态维护
        if (state == S_CALC) {
            singleIteration();
            state = S_SYNC; //计算后可以进行同步
        } else if (state == S_SYNC) {
            this.shouldBreak = combineScores();
            state = S_NORM;  //同步后可以进行正则化表示
        } else if (state == S_NORM) {
            normalizeDeltas();
            state = S_CALC;  //正则化后继续可以进行计算
        } else if (state == S_INIT) {
            initialize();
            state = S_CALC;  //初始化后可以计算
        }
    }

    void normalizeDeltas() {}

    //初始化处理，给定初始的pr值
    private void initialize() {
        this.nextScores = new float[starts.length][];
        Arrays.setAll(nextScores, i -> {
            int size = lengths[i];
            tracker.add(sizeOfFloatArray(size));
            return new float[size];
        });

        tracker.add(sizeOfDoubleArray(partitionSize) << 1);

        double[] partitionRank = new double[partitionSize];
        double initialValue = initialValue();
        if (sourceNodeIds.length == 0) {
            Arrays.fill(partitionRank, initialValue);
        } else {
            Arrays.fill(partitionRank, 0.0);

            long[] partitionSourceNodeIds = LongStream.of(sourceNodeIds)
                .filter(sourceNodeId -> sourceNodeId >= startNode && sourceNodeId < endNode)
                .toArray();

            for (long sourceNodeId : partitionSourceNodeIds) {
                partitionRank[Math.toIntExact(sourceNodeId - this.startNode)] = initialValue;
            }
        }

        this.pageRank = partitionRank;
        this.deltas = Arrays.copyOf(partitionRank, partitionSize);  //给定大小
    }

    double initialValue() {
        return alpha;
    }

    abstract void singleIteration();

    @Override
    public void prepareNormalizeDeltas(double l2Norm) {
        this.l2Norm = l2Norm;
    }

    public void prepareNextIteration(float[][] prevScores) {
        this.prevScores = prevScores;
    }

    //同步
    boolean combineScores() {
        assert prevScores != null;
        assert prevScores.length >= 1;

        int scoreDim = prevScores.length;
        float[][] prevScores = this.prevScores;

        boolean shouldBreak = true;

        int length = prevScores[0].length;
        for (int i = 0; i < length; i++) {
            double sum = 0.0;
            for (int j = 0; j < scoreDim; j++) {
                float[] scores = prevScores[j];
                sum += scores[i];  //对入链分配的值做求和
                scores[i] = 0F;
            }
            double delta = dampingFactor * sum;  //乘以阻尼因子，为了避免Rank Leak【可以理解是等级泄露】和Rank Sink【可以理解是等级沉没】
            if (delta > tolerance) {
                shouldBreak = false;
            }
            pageRank[i] += delta;
            deltas[i] = delta;  //同时把计算的值赋给delta
        }

        return shouldBreak;
    }

    public float[][] nextScores() {
        return nextScores;
    }

    @Override
    public void getPageRankResult(HugeDoubleArray result) {
        result.copyFromArrayIntoSlice(pageRank, startNode, endNode);
    }

    public double[] deltas() { return deltas;}

    @Override
    public boolean partitionIsStable() {
        return shouldBreak;
    }
}
