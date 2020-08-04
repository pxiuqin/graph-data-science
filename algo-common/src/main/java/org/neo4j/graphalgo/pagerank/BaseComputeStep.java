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
        long[] sourceNodeIds,  //起始点
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

    //设置开始节点和节点数量
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
        this.nextScores = new float[starts.length][];  //多少个开始节点来定义下一步得分？因为开始节点数和computeStep个数相同
        Arrays.setAll(nextScores, i -> {
            int size = lengths[i];  //不同起始点对应的长度是不同，这里要针对每个具体的起点来确定【分片中节点的个数】
            tracker.add(sizeOfFloatArray(size));
            return new float[size];
        });

        tracker.add(sizeOfDoubleArray(partitionSize) << 1);

        double[] partitionRank = new double[partitionSize];
        double initialValue = initialValue();  //初始值给定为1-阻尼因子
        if (sourceNodeIds.length == 0) {  //没有给定源节点的话，使用初始化值
            Arrays.fill(partitionRank, initialValue);
        } else {
            Arrays.fill(partitionRank, 0.0);  //先初始化为0

            long[] partitionSourceNodeIds = LongStream.of(sourceNodeIds)
                .filter(sourceNodeId -> sourceNodeId >= startNode && sourceNodeId < endNode)
                .toArray();  //确定分片内节点

            for (long sourceNodeId : partitionSourceNodeIds) {
                partitionRank[Math.toIntExact(sourceNodeId - this.startNode)] = initialValue;  //通过分片中节点的偏移来设置初始值
            }
        }

        this.pageRank = partitionRank;  //分片中节点的多少，记录分片中节点的pr值
        this.deltas = Arrays.copyOf(partitionRank, partitionSize);  //给定大小
    }

    //给定初始值
    double initialValue() {
        return alpha;
    }

    abstract void singleIteration();

    @Override
    public void prepareNormalizeDeltas(double l2Norm) {
        this.l2Norm = l2Norm;
    }

    //给定入链pr值
    public void prepareNextIteration(float[][] prevScores) {
        this.prevScores = prevScores;  //行表示PR计算中初始的ComputeStemp个数，列表示入链
    }

    //合并分值
    boolean combineScores() {
        assert prevScores != null;
        assert prevScores.length >= 1;

        int scoreDim = prevScores.length;  //获取行数，score维度？可以理解成入链的个数
        float[][] prevScores = this.prevScores;

        boolean shouldBreak = true;

        int length = prevScores[0].length;  //分区中的节点数
        for (int i = 0; i < length; i++) {  //每个分区中的节点都需要计算下
            double sum = 0.0;
            for (int j = 0; j < scoreDim; j++) {
                float[] scores = prevScores[j];
                sum += scores[i];  //对入链分配的值做求和【这里入链分配是不准确的，应该理解为计算单元个数或并发线程数】
                scores[i] = 0F;  //？累加后为什么要置0
            }
            double delta = dampingFactor * sum;  //乘以阻尼因子，为了避免Rank Leak【可以理解是等级泄露】和Rank Sink【可以理解是等级沉没】
            if (delta > tolerance) {
                shouldBreak = false;
            }
            pageRank[i] += delta;  //这里为什么要累加？
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
