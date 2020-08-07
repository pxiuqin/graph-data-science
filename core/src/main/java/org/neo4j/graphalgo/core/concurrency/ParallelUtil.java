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
package org.neo4j.graphalgo.core.concurrency;

import org.neo4j.graphalgo.api.BatchNodeIterable;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.loading.HugeParallelGraphImporter;
import org.neo4j.graphalgo.core.utils.BiLongConsumer;
import org.neo4j.graphalgo.core.utils.BitUtil;
import org.neo4j.graphalgo.core.utils.LazyMappingCollection;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterable;
import org.neo4j.graphalgo.utils.ExceptionUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongConsumer;
import java.util.function.Supplier;
import java.util.stream.BaseStream;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.utils.ExceptionUtil.throwIfUnchecked;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public final class ParallelUtil {

    public static final int DEFAULT_BATCH_SIZE = 10_000;

    private static final long DEFAULT_WAIT_TIME_NANOS = 1000;
    private static final long DEFAULT_MAX_NUMBER_OF_RETRIES = (long) 2.5e11; // about 3 days in micros

    // prevent instantiation of factory
    private ParallelUtil() {}

    /**
     * Executes the given function in parallel on the given {@link BaseStream}, using a FJ pool of the requested size.
     * The concurrency value is assumed to already be validated towards the edition limitation.
     */
    public static <T extends BaseStream<?, T>, R> R parallelStream(T data, int concurrency, Function<T, R> fn) {
        ForkJoinPool pool = getFJPoolWithConcurrency(concurrency);
        try {
            return pool.submit(() -> fn.apply(data.parallel())).get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Executes the given function in parallel on the given {@link BaseStream}, using a FJ pool of the requested size.
     * The concurrency value is assumed to already be validated towards the edition limitation.
     */
    public static <T extends BaseStream<?, T>> void parallelStreamConsume(T data, int concurrency, Consumer<T> consumer) {
        parallelStream(data, concurrency, (Function<T, Void>) t -> {
            consumer.accept(t);
            return null;
        });
    }

    public static void parallelForEachNode(Graph graph, int concurrency, LongConsumer consumer) {
        parallelStreamConsume(LongStream.range(0, graph.nodeCount()), concurrency, (stream) -> {
            stream.forEach(consumer);
        });
    }

    /**
     * @return the number of threads required to compute elementCount with the given batchSize
     */
    public static int threadCount(final int batchSize, final int elementCount) {
        return Math.toIntExact(threadCount((long) batchSize, elementCount));
    }

    public static long threadCount(final long batchSize, final long elementCount) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Invalid batch size: " + batchSize);
        }
        if (batchSize >= elementCount) {
            return 1;  //如果batchSize>记录个数直接降级线程为1
        }
        return BitUtil.ceilDiv(elementCount, batchSize);  //给定整除上界
    }

    /**
     * @return a batch size, so that {@code nodeCount} is equally divided by {@code concurrency}
     *     but no smaller than {@code minBatchSize}.
     */
    public static int adjustedBatchSize(
        final int nodeCount,
        int concurrency,
        final int minBatchSize
    ) {
        if (concurrency <= 0) {
            concurrency = nodeCount;
        }
        int targetBatchSize = threadCount(concurrency, nodeCount);
        return Math.max(minBatchSize, targetBatchSize);
    }

    /**
     * @return a batch size, so that {@code nodeCount} is equally divided by {@code concurrency}
     *     but no smaller than {@link #DEFAULT_BATCH_SIZE}.
     * @see #adjustedBatchSize(int, int, int)
     */
    public static int adjustedBatchSize(
        final int nodeCount,
        final int concurrency
    ) {
        return adjustedBatchSize(nodeCount, concurrency, DEFAULT_BATCH_SIZE);
    }

    /**
     * @return a batch size, so that {@code nodeCount} is equally divided by {@code concurrency}
     *     but no smaller than {@link #DEFAULT_BATCH_SIZE}.
     * @see #adjustedBatchSize(int, int, int)
     */
    public static long adjustedBatchSize(
        final long nodeCount,
        int concurrency,
        final long minBatchSize
    ) {
        if (concurrency <= 0) {
            concurrency = (int) Math.min(nodeCount, Integer.MAX_VALUE);  //如果并行度小于0直接按照节点大小来设置
        }
        long targetBatchSize = threadCount(concurrency, nodeCount);
        return Math.max(minBatchSize, targetBatchSize);
    }

    /**
     * @return a batch size, so that {@code nodeCount} is equally divided by {@code concurrency}
     *     but no smaller than {@code minBatchSize} and no larger than {@code maxBatchSize}.
     * @see #adjustedBatchSize(long, int, long)
     */
    public static long adjustedBatchSize(
        final long nodeCount,
        final int concurrency,
        final long minBatchSize,
        final long maxBatchSize
    ) {
        return Math.min(maxBatchSize, adjustedBatchSize(nodeCount, concurrency, minBatchSize));
    }

    /**
     * @return a batch size, that is
     *     1) at least {@code batchSize}
     *     2) a power of two
     *     3) divides {@code nodeCount} into int-sized chunks.
     */
    public static long adjustedBatchSize(final long nodeCount, long batchSize) {
        if (batchSize <= 0L) {
            batchSize = 1L;
        }
        batchSize = BitUtil.nextHighestPowerOfTwo(batchSize);
        while (((nodeCount + batchSize + 1L) / batchSize) > (long) Integer.MAX_VALUE) {
            batchSize = batchSize << 1;
        }
        return batchSize;
    }

    public static boolean canRunInParallel(final ExecutorService executor) {
        return executor != null && !(executor.isShutdown() || executor.isTerminated());
    }

    /**
     * Executes read operations in parallel, based on the given batch size
     * and executor.
     */
    public static <T extends Runnable> void readParallel(
        final int concurrency,
        final int batchSize,
        final BatchNodeIterable idMapping,
        final ExecutorService executor,
        final HugeParallelGraphImporter<T> importer
    ) {

        Collection<PrimitiveLongIterable> iterators =
            idMapping.batchIterables(batchSize);

        int threads = iterators.size();

        if (!canRunInParallel(executor) || threads == 1) {
            long nodeOffset = 0L;
            for (PrimitiveLongIterable iterator : iterators) {
                final T task = importer.newImporter(nodeOffset, iterator);
                task.run();
                nodeOffset += batchSize;
            }
        } else {
            AtomicLong nodeOffset = new AtomicLong();
            Collection<T> importers = LazyMappingCollection.of(
                iterators,
                it -> importer.newImporter(nodeOffset.getAndAdd(batchSize), it)
            );
            runWithConcurrency(concurrency, importers, executor);
        }
    }

    public static void readParallel(
        final int concurrency,
        final long size,
        final ExecutorService executor,
        final BiLongConsumer task
    ) {

        long batchSize = threadCount(concurrency, size);
        if (!canRunInParallel(executor) || concurrency == 1) {
            for (long start = 0L; start < size; start += batchSize) {
                long end = Math.min(size, start + batchSize);
                task.apply(start, end);
            }
        } else {
            Collection<Runnable> threads = new ArrayList<>(concurrency);
            for (long start = 0L; start < size; start += batchSize) {
                long end = Math.min(size, start + batchSize);
                final long finalStart = start;
                threads.add(() -> task.apply(finalStart, end));
            }
            run(threads, executor);
        }
    }

    public static Collection<Runnable> tasks(
        final int concurrency,
        final Supplier<? extends Runnable> newTask
    ) {
        final Collection<Runnable> tasks = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            tasks.add(newTask.get());
        }
        return tasks;
    }

    public static Collection<Runnable> tasks(
        final int concurrency,
        final Function<Integer, ? extends Runnable> newTask
    ) {
        final Collection<Runnable> tasks = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            tasks.add(newTask.apply(i));
        }
        return tasks;
    }

    /**
     * Runs a single task and waits until it's finished.
     */
    public static void run(Runnable task, ExecutorService executor) {
        awaitTermination(Collections.singleton(executor.submit(task)));
    }

    /**
     * Runs a collection of {@link Runnable}s in parallel for their side-effects.
     * The level of parallelism is defined by the given executor.
     * <p>
     * This is similar to {@link ExecutorService#invokeAll(Collection)},
     * except that all Exceptions thrown by any task are chained together.
     */
    public static void run(
        final Collection<? extends Runnable> tasks,
        final ExecutorService executor
    ) {
        run(tasks, executor, null);
    }

    public static void run(
        final Collection<? extends Runnable> tasks,
        final ExecutorService executor,
        final Collection<Future<?>> futures
    ) {
        awaitTermination(run(tasks, true, executor, futures));
    }

    public static Collection<Future<?>> run(
        final Collection<? extends Runnable> tasks,
        final boolean allowSynchronousRun,
        final ExecutorService executor,
        Collection<Future<?>> futures
    ) {

        boolean noExecutor = !canRunInParallel(executor);

        if (allowSynchronousRun && (tasks.size() == 1 || noExecutor)) {
            tasks.forEach(Runnable::run);
            return Collections.emptyList();
        }

        if (noExecutor) {
            throw new IllegalStateException("No running executor provided and synchronous execution is not allowed");
        }

        if (futures == null) {
            futures = new ArrayList<>(tasks.size());
        } else {
            futures.clear();
        }

        for (Runnable task : tasks) {
            futures.add(executor.submit(task));
        }

        return futures;
    }

    public static void run(
        final Collection<? extends Runnable> tasks,
        final Runnable selfTask,
        final ExecutorService executor,
        Collection<Future<?>> futures
    ) {

        if (tasks.isEmpty()) {
            selfTask.run();
            return;
        }

        if (null == executor) {
            tasks.forEach(Runnable::run);
            selfTask.run();
            return;
        }

        if (executor.isShutdown() || executor.isTerminated()) {
            throw new IllegalStateException("Executor is shut down");
        }

        if (futures == null) {
            futures = new ArrayList<>(tasks.size());
        } else {
            futures.clear();
        }

        for (Runnable task : tasks) {
            futures.add(executor.submit(task));
        }

        awaitTermination(futures);
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     *
     * @param concurrency how many tasks should be run simultaneous
     * @param tasks       the tasks to execute
     * @param executor    the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            DEFAULT_WAIT_TIME_NANOS,
            DEFAULT_MAX_NUMBER_OF_RETRIES,
            TerminationFlag.RUNNING_TRUE,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     *
     * @param concurrency how many tasks should be run simultaneous
     * @param tasks       the tasks to execute
     * @param executor    the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long maxRetries,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            DEFAULT_WAIT_TIME_NANOS,
            maxRetries,
            TerminationFlag.RUNNING_TRUE,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     * <p>
     * The provided {@code terminationFlag} is checked before submitting new
     * tasks and if it signals termination, running tasks are cancelled and
     * not-yet-started tasks are abandoned.
     *
     * @param concurrency     how many tasks should be run simultaneous
     * @param tasks           the tasks to execute
     * @param terminationFlag a flag to check periodically if the execution should be terminated
     * @param executor        the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final TerminationFlag terminationFlag,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            DEFAULT_WAIT_TIME_NANOS,
            DEFAULT_MAX_NUMBER_OF_RETRIES,
            terminationFlag,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     * <p>
     * If the pool is full, wait for {@code waitTime} {@code timeUnit}s
     * and retry submitting the tasks indefinitely.
     *
     * @param concurrency how many tasks should be run simultaneous
     * @param tasks       the tasks to execute
     * @param waitTime    how long to wait between retries
     * @param timeUnit    the unit for {@code waitTime}
     * @param executor    the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long waitTime,
        final TimeUnit timeUnit,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            timeUnit.toNanos(waitTime),
            Integer.MAX_VALUE,
            TerminationFlag.RUNNING_TRUE,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     * <p>
     * If the pool is full, wait for {@code waitTime} {@code timeUnit}s
     * and retry submitting the tasks indefinitely.
     * <p>
     * The provided {@code terminationFlag} is checked before submitting new
     * tasks and if it signals termination, running tasks are cancelled and
     * not-yet-started tasks are abandoned.
     *
     * @param concurrency     how many tasks should be run simultaneous
     * @param tasks           the tasks to execute
     * @param waitTime        how long to wait between retries
     * @param timeUnit        the unit for {@code waitTime}
     * @param terminationFlag a flag to check periodically if the execution should be terminated
     * @param executor        the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long waitTime,
        final TimeUnit timeUnit,
        final TerminationFlag terminationFlag,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            timeUnit.toNanos(waitTime),
            Integer.MAX_VALUE,
            terminationFlag,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     * <p>
     * If the pool is full, wait for {@code waitTime} {@code timeUnit}s
     * and retry submitting the tasks at most {@code maxRetries} times.
     *
     * @param concurrency how many tasks should be run simultaneous
     * @param tasks       the tasks to execute
     * @param maxRetries  how many retries when submitting on a full pool before giving up
     * @param waitTime    how long to wait between retries
     * @param timeUnit    the unit for {@code waitTime}
     * @param executor    the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long maxRetries,
        final long waitTime,
        final TimeUnit timeUnit,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            timeUnit.toNanos(waitTime),
            maxRetries,
            TerminationFlag.RUNNING_TRUE,
            executor
        );
    }

    /**
     * Try to run all tasks for their side-effects using at most
     * {@code concurrency} threads at once.
     * <p>
     * If the concurrency is 1 or less, or there is only a single task, or the
     * provided {@link ExecutorService} has terminated the tasks are run
     * sequentially on the calling thread until all tasks are finished or the
     * first Exception is thrown.
     * <p>
     * If the tasks are submitted to the {@code executor}, it may happen that
     * not all tasks are actually executed. If the provided collection creates
     * the tasks lazily upon iteration, not all elements might actually be
     * created.
     * <p>
     * The calling thread will be always blocked during the execution of the tasks
     * and is not available for scheduling purposes. If the calling thread is
     * {@link Thread#interrupt() interrupted} during the execution, running tasks
     * are cancelled and not-yet-started tasks are abandoned.
     * <p>
     * We will try to submit tasks as long as no more than {@code concurrency}
     * are already started and then continue to submit tasks one-by-one, after
     * a previous tasks has finished, so that no more than {@code concurrency}
     * tasks are running in the provided {@code executor}.
     * <p>
     * We do not submit all tasks at-once into the worker queue to avoid creating
     * a large amount of tasks and further support lazily created tasks.
     * We can support thousands, even millions of tasks without resource exhaustion that way.
     * <p>
     * We will try to submit tasks as long as the {@code executor} can
     * directly start new tasks, that is, we want to avoid creating tasks and put
     * them into the waiting queue if it may never be executed afterwards.
     * <p>
     * If the pool is full, wait for {@code waitTime} {@code timeUnit}s
     * and retry submitting the tasks at most {@code maxRetries} times.
     * <p>
     * The provided {@code terminationFlag} is checked before submitting new
     * tasks and if it signals termination, running tasks are cancelled and
     * not-yet-started tasks are abandoned.
     *
     * @param concurrency     how many tasks should be run simultaneous
     * @param tasks           the tasks to execute
     * @param maxRetries      how many retries when submitting on a full pool before giving up
     * @param waitTime        how long to wait between retries
     * @param timeUnit        the unit for {@code waitTime}
     * @param terminationFlag a flag to check periodically if the execution should be terminated
     * @param executor        the executor to submit the tasks to
     */
    public static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long maxRetries,
        final long waitTime,
        final TimeUnit timeUnit,
        final TerminationFlag terminationFlag,
        final ExecutorService executor
    ) {
        runWithConcurrency(
            concurrency,
            tasks,
            timeUnit.toNanos(waitTime),
            maxRetries,
            terminationFlag,
            executor
        );
    }

    private static void runWithConcurrency(
        final int concurrency,
        final Collection<? extends Runnable> tasks,
        final long waitNanos,
        final long maxWaitRetries,
        final TerminationFlag terminationFlag,
        final ExecutorService executor
    ) {
        if (!canRunInParallel(executor) || concurrency <= 1) {
            for (Runnable task : tasks) {
                terminationFlag.assertRunning();
                task.run();
            }
            return;
        }

        CompletionService completionService =
            new CompletionService(executor, concurrency);

        PushbackIterator<Runnable> ts =
            new PushbackIterator<>(tasks.iterator());

        Throwable error = null;
        // generally assumes that tasks.size is notably larger than concurrency
        try {
            //noinspection StatementWithEmptyBody - add first concurrency tasks
            for (int i = concurrency; i-- > 0
                                      && terminationFlag.running()
                                      && completionService.trySubmit(ts); )
                ;

            terminationFlag.assertRunning();

            // submit all remaining tasks
            int tries = 0;
            while (ts.hasNext()) {
                if (completionService.hasTasks()) {
                    try {
                        completionService.awaitNext();
                    } catch (ExecutionException e) {
                        error = ExceptionUtil.chain(error, e.getCause());
                    } catch (CancellationException ignore) {
                    }
                }

                terminationFlag.assertRunning();

                if (!completionService.trySubmit(ts) && !completionService.hasTasks()) {
                    if (++tries >= maxWaitRetries) {
                        throw new IllegalThreadStateException(formatWithLocale(
                            "Attempted to submit tasks for %d times with a %d nanosecond delay (%d milliseconds) between each attempt, but ran out of time",
                            tries,
                            waitNanos,
                            TimeUnit.NANOSECONDS.toMillis(waitNanos)
                        ));
                    }
                    LockSupport.parkNanos(waitNanos);
                }
            }

            // wait for all tasks to finish
            while (completionService.hasTasks()) {
                terminationFlag.assertRunning();
                try {
                    completionService.awaitNext();
                } catch (ExecutionException e) {
                    error = ExceptionUtil.chain(error, e.getCause());
                } catch (CancellationException ignore) {
                }
            }
        } catch (InterruptedException e) {
            error = error == null ? e : ExceptionUtil.chain(e, error);
        } finally {
            finishRunWithConcurrency(completionService, error);
        }
    }

    private static void finishRunWithConcurrency(
        final CompletionService completionService,
        final Throwable error
    ) {
        // cancel all regardless of done flag because we could have aborted
        // from the termination flag
        completionService.cancelAll();
        if (error != null) {
            throwIfUnchecked(error);
            throw new RuntimeException(error);
        }
    }

    public static void awaitTermination(final Collection<Future<?>> futures) {
        boolean done = false;
        Throwable error = null;
        try {
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException ee) {
                    final Throwable cause = ee.getCause();
                    if (error != cause) {
                        error = ExceptionUtil.chain(error, cause);
                    }
                } catch (CancellationException ignore) {
                }
            }
            done = true;
        } catch (InterruptedException e) {
            error = ExceptionUtil.chain(e, error);
        } finally {
            if (!done) {
                for (final Future<?> future : futures) {
                    future.cancel(false);
                }
            }
        }
        if (error != null) {
            throwIfUnchecked(error);
            throw new RuntimeException(error);
        }
    }

    public static void awaitTerminations(final Queue<Future<?>> futures) {
        boolean done = false;
        Throwable error = null;
        try {
            while (!futures.isEmpty()) {
                try {
                    futures.poll().get();
                } catch (ExecutionException ee) {
                    error = ExceptionUtil.chain(error, ee.getCause());
                } catch (CancellationException ignore) {
                }
            }
            done = true;
        } catch (InterruptedException e) {
            error = ExceptionUtil.chain(e, error);
        } finally {
            if (!done) {
                for (final Future<?> future : futures) {
                    future.cancel(false);
                }
            }
        }
        if (error != null) {
            throwIfUnchecked(error);
            throw new RuntimeException(error);
        }
    }

    /**
     * Copied from {@link java.util.concurrent.ExecutorCompletionService}
     * and adapted to reduce indirection.
     * Does not support {@link java.util.concurrent.ForkJoinPool} as backing executor.
     */
    private static final class CompletionService {
        private final Executor executor;
        private final ThreadPoolExecutor pool;
        private final int availableConcurrency;
        private final Set<Future<Void>> running;
        private final BlockingQueue<Future<Void>> completionQueue;

        private class QueueingFuture extends FutureTask<Void> {
            QueueingFuture(final Runnable runnable) {
                super(runnable, null);
                running.add(this);
            }

            @Override
            protected void done() {
                running.remove(this);
                if (!isCancelled()) {
                    //noinspection StatementWithEmptyBody - spin-wait on free slot
                    while (!completionQueue.offer(this)) ;
                }
            }
        }

        CompletionService(final ExecutorService executor, final int targetConcurrency) {
            if (!canRunInParallel(executor)) {
                throw new IllegalArgumentException(
                    "executor already terminated or not usable");
            }
            if (executor instanceof ThreadPoolExecutor) {
                pool = (ThreadPoolExecutor) executor;
                availableConcurrency = pool.getCorePoolSize();
                int capacity = Math.max(targetConcurrency, availableConcurrency) + 1;
                completionQueue = new ArrayBlockingQueue<>(capacity);
            } else {
                pool = null;
                availableConcurrency = Integer.MAX_VALUE;
                completionQueue = new LinkedBlockingQueue<>();
            }

            this.executor = executor;
            this.running = Collections.newSetFromMap(new ConcurrentHashMap<>());
        }

        boolean trySubmit(final PushbackIterator<Runnable> tasks) {
            if (tasks.hasNext()) {
                Runnable next = tasks.next();
                if (submit(next)) {
                    return true;
                }
                tasks.pushBack(next);
            }
            return false;
        }

        boolean submit(final Runnable task) {
            Objects.requireNonNull(task);
            if (canSubmit()) {
                executor.execute(new QueueingFuture(task));
                return true;
            }
            return false;
        }

        boolean hasTasks() {
            return !(running.isEmpty() && completionQueue.isEmpty());
        }

        void awaitNext() throws InterruptedException, ExecutionException {
            completionQueue.take().get();
        }

        void cancelAll() {
            stopFuturesAndStopScheduling(running);
            stopFutures(completionQueue);
        }

        private boolean canSubmit() {
            return pool == null || pool.getActiveCount() < availableConcurrency;
        }

        private void stopFutures(final Collection<Future<Void>> futures) {
            for (Future<Void> future : futures) {
                future.cancel(false);
            }
            futures.clear();
        }

        private void stopFuturesAndStopScheduling(final Collection<Future<Void>> futures) {
            if (pool == null) {
                stopFutures(futures);
                return;
            }
            for (Future<Void> future : futures) {
                if (future instanceof Runnable) {
                    pool.remove((Runnable) future);
                }
                future.cancel(false);
            }
            futures.clear();
            pool.purge();
        }
    }

    private static final class PushbackIterator<T> implements Iterator<T> {
        private final Iterator<? extends T> delegate;
        private T pushedElement;

        private PushbackIterator(final Iterator<? extends T> delegate) {
            this.delegate = delegate;
        }

        @Override
        public boolean hasNext() {
            return pushedElement != null || delegate.hasNext();
        }

        @Override
        public T next() {
            T el;
            if ((el = pushedElement) != null) {
                pushedElement = null;
            } else {
                el = delegate.next();
            }
            return el;
        }

        void pushBack(final T element) {
            if (pushedElement != null) {
                throw new IllegalArgumentException("Cannot push back twice");
            }
            pushedElement = element;
        }
    }

    private static ForkJoinPool getFJPoolWithConcurrency(int concurrency) {
        return new ForkJoinPool(concurrency);
    }
}
