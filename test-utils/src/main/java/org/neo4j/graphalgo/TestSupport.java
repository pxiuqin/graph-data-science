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
package org.neo4j.graphalgo;

import org.hamcrest.BaseMatcher;
import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.canonization.CanonicalAdjacencyMatrix;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.concurrency.ParallelUtil;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.extension.GdlSupportExtension;
import org.neo4j.graphalgo.extension.IdFunction;
import org.neo4j.graphalgo.extension.TestGraph;
import org.neo4j.graphalgo.gdl.GdlFactory;
import org.neo4j.graphalgo.gdl.ImmutableGraphCreateFromGdlConfig;
import org.neo4j.graphdb.TransactionTerminatedException;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.api.exceptions.Status;
import org.neo4j.kernel.impl.coreapi.InternalTransaction;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.Orientation.NATURAL;
import static org.neo4j.graphalgo.Orientation.REVERSE;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public final class TestSupport {

    public enum FactoryType {
        NATIVE, CYPHER
    }

    private TestSupport() {}

    @Retention(RetentionPolicy.RUNTIME)
    @ParameterizedTest
    @MethodSource("org.neo4j.graphalgo.TestSupport#allFactoryTypes")
    public @interface AllGraphStoreFactoryTypesTest {}

    public static Stream<FactoryType> allFactoryTypes() {
        return Stream.of(FactoryType.NATIVE, FactoryType.CYPHER);
    }

    public static Stream<Orientation> allDirectedProjections() {
        return Stream.of(NATURAL, REVERSE);
    }

    public static <T> Supplier<Stream<Arguments>> toArguments(Supplier<Stream<T>> fn) {
        return () -> fn.get().map(Arguments::of);
    }

    @SafeVarargs
    public static Stream<Arguments> crossArguments(Supplier<Stream<Arguments>> firstFn, Supplier<Stream<Arguments>>... otherFns) {
        return Arrays
                .stream(otherFns)
                .reduce(firstFn, (l, r) -> () -> crossArguments(l, r))
                .get();
    }

    public static Stream<Arguments> crossArguments(Supplier<Stream<Arguments>> leftFn, Supplier<Stream<Arguments>> rightFn) {
        return leftFn.get().flatMap(leftArgs ->
                rightFn.get().map(rightArgs -> {
                    Collection<Object> leftObjects = new ArrayList<>(Arrays.asList(leftArgs.get()));
                    leftObjects.addAll(new ArrayList<>(Arrays.asList(rightArgs.get())));
                    return Arguments.of(leftObjects.toArray());
                }));
    }

    public static TestGraph fromGdl(String gdl) {
        return fromGdl(gdl, NATURAL, "graph");
    }

    public static TestGraph fromGdl(String gdl, String name) {
        return fromGdl(gdl, NATURAL, name);
    }

    public static TestGraph fromGdl(String gdl, Orientation orientation) {
        return fromGdl(gdl, orientation, "graph");
    }

    public static TestGraph fromGdl(String gdl, Orientation orientation, String name) {
        Objects.requireNonNull(gdl);

        var config = ImmutableGraphCreateFromGdlConfig.builder()
            .gdlGraph(gdl)
            .graphName("graph")
            .orientation(orientation)
            .build();

        var gdlFactory = GdlFactory.of(config, GdlSupportExtension.DATABASE_ID);

        return new TestGraph(gdlFactory.build().graphStore().getUnion(), gdlFactory::nodeId, name);
    }

    public static long[][] ids(IdFunction idFunction, String[][] variables) {
        return Arrays.stream(variables).map(vs -> ids(idFunction, vs)).toArray(long[][]::new);
    }

    public static long[] ids(IdFunction idFunction, String... variables) {
        return Arrays.stream(variables).mapToLong(idFunction::of).toArray();
    }

    public static void assertLongValues(TestGraph graph, Function<Long, Long> actualValues, Map<String, Long> expectedValues) {
        expectedValues.forEach((variable, expectedValue) -> {
            Long actualValue = actualValues.apply(graph.toMappedNodeId(variable));
            assertEquals(
                expectedValue,
                actualValue,
                formatWithLocale(
                    "Values do not match for variable %s. Expected %s, got %s.",
                    variable,
                    expectedValue.toString(),
                    actualValue.toString()
                ));
        });
    }

    public static void assertDoubleValues(TestGraph graph, Function<Long, Double> actualValues, Map<String, Double> expectedValues, double delta) {
        expectedValues.forEach((variable, expectedValue) -> {
            Double actualValue = actualValues.apply(graph.toMappedNodeId(variable));
            assertEquals(
                expectedValue,
                actualValue,
                delta,
                formatWithLocale(
                    "Values do not match for variable %s. Expected %s, got %s.",
                    variable,
                    expectedValue.toString(),
                    actualValue.toString()
                ));
        });
    }


    public static void assertGraphEquals(Graph expected, Graph actual) {
        Assertions.assertEquals(expected.nodeCount(), actual.nodeCount(), "Node counts do not match.");
        // TODO: we cannot check this right now, because the relationshhip counts depends on how the graph has been loaded for HugeGraph
//        Assertions.assertEquals(expected.relationshipCount(), actual.relationshipCount(), "Relationship counts to not match.");
        Assertions.assertEquals(CanonicalAdjacencyMatrix.canonicalize(expected), CanonicalAdjacencyMatrix.canonicalize(actual));
    }

    /**
     * Checks if exactly one of the given expected graphs matches the actual graph.
     */
    public static void assertGraphEquals(Collection<Graph> expectedGraphs, Graph actual) {
        List<String> expectedCanonicalized = expectedGraphs.stream().map(CanonicalAdjacencyMatrix::canonicalize).collect(Collectors.toList());
        String actualCanonicalized = CanonicalAdjacencyMatrix.canonicalize(actual);

        boolean equals = expectedCanonicalized
            .stream()
            .map(expected -> expected.equals(actualCanonicalized))
            .reduce(Boolean::logicalXor)
            .orElse(false);

        String message = formatWithLocale(
            "None of the given graphs matches the actual one.%nActual:%n%s%nExpected:%n%s",
            actualCanonicalized,
            String.join("\n\n", expectedCanonicalized)
        );

        assertTrue(equals, message);
    }

    public static void assertMemoryEstimation(
        Supplier<MemoryEstimation> actualMemoryEstimation,
        long nodeCount,
        int concurrency,
        long expectedMinBytes,
        long expectedMaxBytes
    ) {
        assertMemoryEstimation(
            actualMemoryEstimation,
            GraphDimensions.of(nodeCount),
            concurrency,
            expectedMinBytes,
            expectedMaxBytes
        );
    }

    public static void assertMemoryEstimation(
        Supplier<MemoryEstimation> actualMemoryEstimation,
        GraphDimensions dimensions,
        int concurrency,
        long expectedMinBytes,
        long expectedMaxBytes
    ) {
        var actual = actualMemoryEstimation.get().estimate(dimensions, concurrency).memoryUsage();

        assertEquals(expectedMinBytes, actual.min);
        assertEquals(expectedMaxBytes, actual.max);
    }

    public static <K, V> Matcher<Map<K, ? extends V>> mapEquals(Map<K, V> expected) {
        return new BaseMatcher<>() {
            @Override
            public boolean matches(Object actual) {
                if (!(actual instanceof Map)) {
                    return false;
                }
                Map<K, V> actualMap = (Map<K, V>) actual;
                if (!actualMap.keySet().equals(expected.keySet())) {
                    return false;
                }
                for (Object key : expected.keySet()) {
                    if (!expected.get(key).equals(actualMap.get(key))) {
                        return false;
                    }
                }
                return true;
            }

            @Override
            public void describeTo(Description description) {
                description.appendText(expected.toString());
            }
        };
    }

    /**
     * This method assumes that the given algorithm calls {@link Algorithm#assertRunning()} at least once.
     * When called, the algorithm will sleep for {@code sleepMillis} milliseconds before it checks the transaction state.
     * A second thread will terminate the transaction during the sleep interval.
     */
    public static void assertAlgorithmTermination(
        GraphDatabaseAPI db,
        Algorithm<?, ?> algorithm,
        Consumer<Algorithm<?, ?>> algoConsumer,
        long sleepMillis
    ) {
        assert sleepMillis >= 100 && sleepMillis <= 10_000;

        var timeoutTx = db.beginTx(10, TimeUnit.SECONDS);
        KernelTransaction kernelTx = ((InternalTransaction) timeoutTx).kernelTransaction();
        algorithm.withTerminationFlag(new TestTerminationFlag(kernelTx, sleepMillis));

        Runnable algorithmThread = () -> {
            try {
                algoConsumer.accept(algorithm);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };

        Runnable interruptingThread = () -> {
            try {
                Thread.sleep(sleepMillis / 2);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            kernelTx.markForTermination(Status.Transaction.TransactionMarkedAsFailed);
        };

        assertThrows(
            TransactionTerminatedException.class,
            () -> {
                try {
                    ParallelUtil.run(Arrays.asList(algorithmThread, interruptingThread), Pools.DEFAULT);
                } catch (RuntimeException e) {
                    throw e.getCause();
                }
            }
        );
    }

    public static void assertTransactionTermination(Executable executable) {
        TransactionTerminatedException exception = assertThrows(
            TransactionTerminatedException.class,
            executable
        );

        assertEquals(Status.Transaction.Terminated, exception.status());
    }

    public static String getCypherAggregation(String aggregation, String property) {
        String cypherAggregation;
        switch (Aggregation.lookup(aggregation)) {
            case SINGLE:
                cypherAggregation = "head(collect(%s))";
                break;
            case SUM:
                cypherAggregation = "sum(%s)";
                break;
            case MIN:
                cypherAggregation = "min(%s)";
                break;
            case MAX:
                cypherAggregation = "max(%s)";
                break;
            case COUNT:
                cypherAggregation = "count(%s)";
                break;
            default:
                cypherAggregation = "%s";
                break;
        }
        return formatWithLocale(cypherAggregation, property);
    }
}
