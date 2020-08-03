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

import com.carrotsearch.hppc.IntArrayList;
import com.carrotsearch.hppc.LongArrayList;
import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.IdMapping;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.core.utils.partition.Partition;
import org.neo4j.graphalgo.core.utils.partition.PartitionUtils;
import org.neo4j.graphalgo.result.CentralityResult;
import org.neo4j.logging.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.core.utils.BitUtil.ceilDiv;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.humanReadable;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfDoubleArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfInstance;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfIntArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfLongArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfObjectArray;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

/**
 * Partition based parallel Page Rank based on
 * "An Efficient Partition-Based Parallel PageRank Algorithm" [1]
 * <p>
 * Each partition thread has its local array of only the nodes that it is responsible for,
 * not for all nodes. Combined, all partitions hold all page rank scores for every node once.
 * Instead of writing partition files and transferring them across the network
 * (as done in the paper since they were concerned with parallelising across multiple nodes),
 * we use integer arrays to write the results to.
 * The actual score is upscaled from a double to an integer by multiplying it with {@code 100_000}.
 * <p>To avoid contention by writing to a shared array, we partition the result array.
 * During execution, the scores arrays
 * are shaped like this:</p>
 * <pre>
 *     [ executing partition ] -&gt; [ calculated partition ] -&gt; [ local page rank scores ]
 * </pre>
 * <p>
 * Each single partition writes in a partitioned array, calculation the scores
 * for every receiving partition. A single partition only sees:
 * <pre>
 *     [ calculated partition ] -&gt; [ local page rank scores ]
 * </pre>
 * <p>The coordinating thread then builds the transpose of all written partitions from every partition:</p>
 * <pre>
 *     [ calculated partition ] -&gt; [ executing partition ] -&gt; [ local page rank scores ]
 * </pre>
 * <p>This step does not happen in parallel, but does not involve extensive copying.
 * The local page rank scores needn't be copied, only the partitioning arrays.
 * All in all, {@code concurrency^2} array element reads and assignments have to
 * be performed.</p>
 * <p>For the next iteration, every partition first updates its scores, in parallel.
 * A single partition now sees:</p>
 * <pre>
 *     [ executing partition ] -&gt; [ local page rank scores ]
 * </pre>
 * <p>That is, a list of all calculated scores for it self, grouped by the partition that
 * calculated these scores.
 * This means, most of the synchronization happens in parallel, too.</p>
 * <p>
 * Partitioning is not done by number of nodes but by the accumulated degree –
 * as described in "Fast Parallel PageRank: A Linear System Approach" [2].
 * Every partition should have about the same number of relationships to operate on.
 * This is done to avoid having one partition with super nodes and instead have
 * all partitions run in approximately equal time.
 * Smaller partitions are merged down until we have at most {@code concurrency} partitions,
 * in order to batch partitions and keep the number of threads in use predictable/configurable.
 * </p>
 * <p>
 * [1]: <a href="http://delab.csd.auth.gr/~dimitris/courses/ir_spring06/page_rank_computing/01531136.pdf">An Efficient Partition-Based Parallel PageRank Algorithm</a><br>
 * [2]: <a href="https://www.cs.purdue.edu/homes/dgleich/publications/gleich2004-parallel.pdf">Fast Parallel PageRank: A Linear System Approach</a>
 * </p>
 */
public class PageRank extends Algorithm<PageRank, PageRank> {

    public static final double DEFAULT_WEIGHT = 1.0D;
    public static final Double DEFAULT_TOLERANCE = 0.0000001D;

    private final ExecutorService executor;
    private final int concurrency;
    private final int batchSize;
    private final AllocationTracker tracker;
    private final IdMapping idMapping;
    private final double dampingFactor;
    private final int maxIterations;
    private int ranIterations;
    private boolean didConverge;
    private final double toleranceValue;
    private final Graph graph;
    private final LongStream sourceNodeIds;
    private final PageRankVariant pageRankVariant;

    private ComputeSteps computeSteps;   //并行计算单元

    private final HugeDoubleArray result;  //每个计算单元的计算结果

    /**
     * Parallel Page Rank implementation.
     * Whether the algorithm actually runs in parallel depends on the given
     * executor and batchSize.
     * 1.并行化处理使用Partition的格式（src_id,out_degress,[dst_id1,dst_id2,...,dst_idn]）
     * 2.迭代计算把当前src_id的pr值传递到dst_id
     * 3.累加dst_id的score
     * 4.最终pr=(1-a)/T+a*V_score
     * 
     */
    PageRank(
        Graph graph,
        PageRankVariant pageRankVariant,
        LongStream sourceNodeIds,
        PageRankBaseConfig algoConfig,
        int concurrency,
        ExecutorService executor,
        int batchSize,
        ProgressLogger progressLogger,
        AllocationTracker tracker
    ) {
        assert algoConfig.maxIterations() >= 1;
        this.executor = executor;
        this.concurrency = concurrency;
        this.batchSize = batchSize;
        this.tracker = tracker;
        this.idMapping = graph;
        this.graph = graph;
        this.dampingFactor = algoConfig.dampingFactor();
        this.maxIterations = algoConfig.maxIterations();
        this.ranIterations = 0;
        this.didConverge = false;
        this.toleranceValue = algoConfig.tolerance();
        this.sourceNodeIds = sourceNodeIds;
        this.pageRankVariant = pageRankVariant;
        this.result = HugeDoubleArray.newArray(graph.nodeCount(), tracker);
        this.progressLogger = progressLogger;
    }

    public int iterations() {
        return ranIterations;
    }

    public boolean didConverge() {
        return didConverge;
    }

    public double dampingFactor() {
        return dampingFactor;
    }

    /**
     * compute pageRank for n iterations
     */
    @Override
    public PageRank compute() {
        getProgressLogger().logMessage(":: Start");

        initializeSteps();
        computeSteps.run(maxIterations);   //基于迭代次数来计算
        computeSteps.mergeResults();

        getProgressLogger().logMessage(":: Finished");
        return this;
    }

    public CentralityResult result() {
        return new CentralityResult(result);
    }

    // we cannot do this in the constructor anymore since
    // we want to allow the user to provide a log instance
    private void initializeSteps() {
        if (computeSteps != null) {
            return;
        }

        //确定分片
        List<Partition> partitions = PartitionUtils.degreePartition(graph, adjustBatchSize(batchSize));

        ExecutorService executor = ParallelUtil.canRunInParallel(this.executor)
                ? this.executor : null;

        computeSteps = createComputeSteps(
                concurrency,
                idMapping.nodeCount(),
                dampingFactor,
                sourceNodeIds.map(graph::toMappedNodeId).filter(mappedId -> mappedId != -1L).toArray(),
                partitions,
                executor);
    }

    //调整分片数
    private int adjustBatchSize(int batchSize) {
        if (batchSize == 0) {
            return Partition.MAX_NODE_COUNT;
        }

        // multiply batchsize by average degree, so that the resulting
        // partitions are sized closer to the provided batchSize
        long averageDegree = Math.max(1, ceilDiv(graph.relationshipCount(), graph.nodeCount()));
        long degreeBatchSize = averageDegree * batchSize;

        return (int) Math.min(degreeBatchSize, Partition.MAX_NODE_COUNT);
    }

    //创建计算单元
    private ComputeSteps createComputeSteps(
            int concurrency,
            long nodeCount,
            double dampingFactor,
            long[] sourceNodeIds,
            List<Partition> partitions,
            ExecutorService pool) {
        concurrency = findIdealConcurrency(nodeCount, partitions, concurrency, progressLogger.getLog());
        final int expectedParallelism = Math.min(
                concurrency,
                partitions.size());

        List<ComputeStep> computeSteps = new ArrayList<>(expectedParallelism);  //并行数
        LongArrayList starts = new LongArrayList(expectedParallelism);
        IntArrayList lengths = new IntArrayList(expectedParallelism);
        int partitionsPerThread = ParallelUtil.threadCount(
                concurrency + 1,
                partitions.size());
        Iterator<Partition> parts = partitions.iterator();

        DegreeComputer degreeComputer = pageRankVariant.degreeComputer(graph);
        DegreeCache degreeCache = degreeComputer.degree(pool, concurrency, tracker);

        //对每个分片进行处理
        while (parts.hasNext()) {
            Partition partition = parts.next();  //获取一个分片
            int partitionSize = (int) partition.nodeCount;  //当前分片的节点个数
            long start = partition.startNode;  //当前分配的开始节点
            int i = 1;
            while (parts.hasNext()
                   && i < partitionsPerThread  //每个线程分片数量
                   && partition.fits(partitionSize)) {  //判断满足条件
                partition = parts.next();
                partitionSize += partition.nodeCount;  //处理的数量
                ++i;
            }

            starts.add(start);  //开始节点保存
            lengths.add(partitionSize);

            //添加一个可计算单元，使用PageRankVariant来创建计算单元
            computeSteps.add(pageRankVariant.createComputeStep(
                    dampingFactor,
                    toleranceValue,
                    sourceNodeIds,
                    graph,
                    tracker,
                    partitionSize,
                    start,
                    degreeCache,
                    nodeCount,
                    progressLogger
            ));
        }

        long[] startArray = starts.toArray();
        int[] lengthArray = lengths.toArray();
        for (ComputeStep computeStep : computeSteps) {
            computeStep.setStarts(startArray, lengthArray);  //每个计算单元都给定开始节点数组和长度数组
        }
        return new ComputeSteps(tracker, computeSteps, concurrency, pool);
    }

    private static int findIdealConcurrency(
            long nodeCount,
            List<Partition> partitions,
            int concurrency,
            Log log) {
        if (concurrency <= 0) {
            concurrency = partitions.size();
        }

        if (log != null && log.isDebugEnabled()) {
            log.debug(
                    "PageRank: nodes=%d, concurrency=%d, available memory=%s, estimated memory usage: %s",
                    nodeCount,
                    concurrency,
                    humanReadable(availableMemory()),
                    humanReadable(memoryUsageFor(concurrency, partitions))
            );
        }

        int maxConcurrency = maxConcurrencyByMemory(
                nodeCount,
                concurrency,
                availableMemory(),
                partitions);

        if (concurrency > maxConcurrency) {
            if (log != null) {
                long required = memoryUsageFor(concurrency, partitions);
                long newRequired = memoryUsageFor(maxConcurrency, partitions);
                long available = availableMemory();
                log.warn(
                        "Requested concurrency of %d would require %s Heap but only %s are available, Page Rank will be throttled to a concurrency of %d to use only %s Heap.",
                        concurrency,
                        humanReadable(required),
                        humanReadable(available),
                        maxConcurrency,
                        humanReadable(newRequired)
                );
            }
            concurrency = maxConcurrency;
        }
        return concurrency;
    }

    private static int maxConcurrencyByMemory(
            long nodeCount,
            int concurrency,
            long availableBytes,
            List<Partition> partitions) {
        int newConcurrency = concurrency;

        long memoryUsage = memoryUsageFor(newConcurrency, partitions);
        while (memoryUsage > availableBytes) {
            long perThread = estimateMemoryUsagePerThread(nodeCount, concurrency);
            long overflow = memoryUsage - availableBytes;
            newConcurrency -= (int) Math.ceil((double) overflow / (double) perThread);

            memoryUsage = memoryUsageFor(newConcurrency, partitions);
        }

        if (newConcurrency < 1) {
            throw new IllegalStateException(
                formatWithLocale(
                    "Requested concurrency of %d would require %s Heap but only %s are available. Page Rank needs at least %d Heap in order to run.",
                    concurrency,
                    humanReadable(memoryUsageFor(concurrency, partitions)),
                    humanReadable(memoryUsageFor(1, partitions))
                )
            );
        }

        return newConcurrency;
    }

    private static long availableMemory() {
        // TODO: run gc first to free up memory?
        Runtime rt = Runtime.getRuntime();

        long max = rt.maxMemory(); // max allocated 内存分配上线xmx
        long total = rt.totalMemory(); // currently allocated 当前分配的xms
        long free = rt.freeMemory(); // unused portion of currently allocated 当前分配的还未用的

        return max - total + free;  //如此来计算当前可用内存总量
    }

    private static long estimateMemoryUsagePerThread(long nodeCount, int concurrency) {
        int nodesPerThread = (int) Math.ceil((double) nodeCount / (double) concurrency);
        long partitions = sizeOfIntArray(nodesPerThread) * (long) concurrency;
        return sizeOfInstance(BaseComputeStep.class) + partitions;
    }

    private static long memoryUsageFor(
            int concurrency,
            List<Partition> partitions) {
        long perThreadUsage = 0L;
        long sharedUsage = 0L;
        int stepSize = 0;
        int partitionsPerThread = ParallelUtil.threadCount(concurrency + 1, partitions.size());
        Iterator<Partition> parts = partitions.iterator();

        while (parts.hasNext()) {
            Partition partition = parts.next();
            int partitionCount = (int) partition.nodeCount;
            int i = 1;
            while (parts.hasNext()
                   && i < partitionsPerThread
                   && partition.fits(partitionCount)) {
                partition = parts.next();
                partitionCount += partition.nodeCount;
                ++i;
            }
            stepSize++;
            sharedUsage += (sizeOfDoubleArray(partitionCount) << 1);
            perThreadUsage += sizeOfIntArray(partitionCount);
        }

        perThreadUsage *= stepSize;
        perThreadUsage += sizeOfInstance(BaseComputeStep.class);
        perThreadUsage += sizeOfObjectArray(stepSize);

        sharedUsage += sizeOfInstance(ComputeSteps.class);
        sharedUsage += sizeOfLongArray(stepSize) << 1;

        return sharedUsage + perThreadUsage;
    }

    @Override
    public PageRank me() {
        return this;
    }

    @Override
    public void release() {
        computeSteps.release();
    }

    //并行计算单元组装
    public final class ComputeSteps {
        private List<ComputeStep> steps;
        private final ExecutorService pool;
        private float[][][] scores;  //等分最终构建起一个3维数组
        private final int concurrency;

        private ComputeSteps(
                AllocationTracker tracker,
                List<ComputeStep> steps,
                int concurrency,
                ExecutorService pool) {
            this.concurrency = concurrency;
            assert !steps.isEmpty();
            this.steps = steps;
            this.pool = pool;
            int stepSize = steps.size();
            scores = new float[stepSize][stepSize][];  //？
            if (AllocationTracker.isTracking(tracker)) {
                tracker.add((stepSize + 1) * sizeOfObjectArray(stepSize));
            }
        }

        void mergeResults() {
            for (ComputeStep step : steps) {
                step.getPageRankResult(result);
            }
        }

        private void run(int iterations) {
            didConverge = false;  //标识是否停止迭代
            ParallelUtil.runWithConcurrency(concurrency, steps, terminationFlag, pool);
            for (ranIterations = 0; ranIterations < iterations && !didConverge; ranIterations++) {
                getProgressLogger().logMessage(formatWithLocale(":: Iteration %d :: Start", ranIterations + 1));

                // calculate scores
                ParallelUtil.runWithConcurrency(concurrency, steps, terminationFlag, pool);

                // sync scores
                synchronizeScores();
                ParallelUtil.runWithConcurrency(concurrency, steps, terminationFlag, pool);
                didConverge = checkTolerance();

                // normalize deltas
                normalizeDeltas();
                ParallelUtil.runWithConcurrency(concurrency, steps, terminationFlag, pool);

                if ((ranIterations < iterations - 1) && !didConverge) {
                    getProgressLogger().reset(graph.relationshipCount());
                }

                getProgressLogger().logMessage(formatWithLocale(":: Iteration %d :: Finished", ranIterations + 1));
            }
        }

        private boolean checkTolerance() {
            return steps.stream().allMatch(ComputeStep::partitionIsStable);
        }

        private void normalizeDeltas() {
            double l2Norm = computeNorm();

            for (ComputeStep step : steps) {
                step.prepareNormalizeDeltas(l2Norm);
            }
        }

        private double computeNorm() {
            double l2Norm = 0.0;
            for (ComputeStep step : steps) {
                double[] deltas = step.deltas();
                l2Norm += ParallelUtil.parallelStream(
                        Arrays.stream(deltas),
                        concurrency,
                        (stream) -> stream.map(score -> score * score).sum());  //并行计算平方和
            }
            l2Norm = Math.sqrt(l2Norm);   //平方和在开平方，计算L2范数
            l2Norm = l2Norm < 0 ? 1 : l2Norm;
            return l2Norm;
        }

        private void synchronizeScores() {
            int stepSize = steps.size();
            float[][][] scores = this.scores;
            int i;

            //每个并行步分别同步下相关得分
            for (i = 0; i < stepSize; i++) {
                synchronizeScores(steps.get(i), i, scores);
            }
        }

        //同步得分
        private void synchronizeScores(ComputeStep step, int idx, float[][][] scores) {
            step.prepareNextIteration(scores[idx]);
            float[][] nextScores = step.nextScores();
            for (int j = 0, len = nextScores.length; j < len; j++) {
                scores[j][idx] = nextScores[j];  //给出一个计算单元中的得分值【同步的目的就是把入链值转换到出链值】
            }
        }

        private void release() {
            if (AllocationTracker.isTracking(tracker)) {
                tracker.remove((scores.length + 1) * sizeOfObjectArray(scores.length));
            }
            steps.clear();
            steps = null;
            scores = null;
        }
    }
}
