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

import com.carrotsearch.hppc.IntArrayList;
import com.carrotsearch.hppc.IntObjectHashMap;
import com.carrotsearch.hppc.IntObjectMap;
import com.carrotsearch.hppc.cursors.IntObjectCursor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.TestLog;
import org.neo4j.graphalgo.TestProgressLogger;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeLongArray;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.Inject;
import org.neo4j.graphalgo.extension.TestGraph;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.TestSupport.assertMemoryEstimation;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

@GdlExtension
class LabelPropagationTest {

    private static final LabelPropagationStreamConfig DEFAULT_CONFIG = LabelPropagationStreamConfig.of(
        "",
        Optional.empty(),
        Optional.empty(),
        CypherMapWrapper.empty()
    );

    @GdlGraph
    private static final String GRAPH =
        "CREATE" +
        "  (nAlice:User   {seedId: 2})" +
        ", (nBridget:User {seedId: 3})" +
        ", (nCharles:User {seedId: 4})" +
        ", (nDoug:User    {seedId: 3})" +
        ", (nMark:User    {seedId: 4})" +
        ", (nMichael:User {seedId: 2})" +
        ", (nAlice)-[:FOLLOW]->(nBridget)" +
        ", (nAlice)-[:FOLLOW]->(nCharles)" +
        ", (nMark)-[:FOLLOW]->(nDoug)" +
        ", (nBridget)-[:FOLLOW]->(nMichael)" +
        ", (nDoug)-[:FOLLOW]->(nMark)" +
        ", (nMichael)-[:FOLLOW]->(nAlice)" +
        ", (nAlice)-[:FOLLOW]->(nMichael)" +
        ", (nBridget)-[:FOLLOW]->(nAlice)" +
        ", (nMichael)-[:FOLLOW]->(nBridget)" +
        ", (nCharles)-[:FOLLOW]->(nDoug)";

    @Inject
    private TestGraph graph;

    @Test
    void shouldUseOriginalNodeIdWhenSeedPropertyIsMissing() {
        LabelPropagation lp = new LabelPropagation(
            graph,
            ImmutableLabelPropagationStreamConfig.builder().maxIterations(1).build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.empty()
        );
        assertArrayEquals(
            new long[]{
                graph.toMappedNodeId("nBridget"),
                graph.toMappedNodeId("nBridget"),
                graph.toMappedNodeId("nDoug"),
                graph.toMappedNodeId("nMark"),
                graph.toMappedNodeId("nMark"),
                graph.toMappedNodeId("nBridget")
            },
            lp.compute().labels().toArray(),
            "Incorrect result assuming initial labels are neo4j id"
        );
    }

    @Test
    void shouldUseSeedProperty() {
        LabelPropagation lp = new LabelPropagation(
            graph,
            ImmutableLabelPropagationStreamConfig
                .builder()
                .seedProperty("seedId")
                .maxIterations(1)
                .build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.empty()
        );

        assertArrayEquals(
            new long[]{
                graph.toMappedNodeId("nCharles"),
                graph.toMappedNodeId("nCharles"),
                graph.toMappedNodeId("nDoug"),
                graph.toMappedNodeId("nMark"),
                graph.toMappedNodeId("nMark"),
                graph.toMappedNodeId("nCharles")
            },
            lp.compute().labels().toArray()
        );
    }

    @Test
    void testSingleThreadClustering() {
        testClustering(graph, 100);
    }

    @Test
    void testMultiThreadClustering() {
        testClustering(graph, 2);
    }

    private void testClustering(Graph graph, int batchSize) {
        for (int i = 0; i < 20; i++) {
            testLPClustering(graph, batchSize);
        }
    }

    private void testLPClustering(Graph graph, int batchSize) {
        LabelPropagation lp = new LabelPropagation(
            graph,
            DEFAULT_CONFIG,
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.empty()
        );
        lp.withBatchSize(batchSize);
        lp.compute();
        HugeLongArray labels = lp.labels();
        assertNotNull(labels);
        IntObjectMap<IntArrayList> cluster = groupByPartitionInt(labels);
        assertNotNull(cluster);

        assertTrue(lp.didConverge());
        assertTrue(2L <= lp.ranIterations(), "expected at least 2 iterations, got " + lp.ranIterations());
        assertEquals(2L, cluster.size());
        for (IntObjectCursor<IntArrayList> cursor : cluster) {
            int[] ids = cursor.value.toArray();
            Arrays.sort(ids);
            if (cursor.key == 0 || cursor.key == 1 || cursor.key == 5) {
                assertArrayEquals(new int[]{0, 1, 5}, ids);
            } else {
                assertArrayEquals(new int[]{2, 3, 4}, ids);
            }
        }
    }

    private static IntObjectMap<IntArrayList> groupByPartitionInt(HugeLongArray labels) {
        if (labels == null) {
            return null;
        }
        IntObjectMap<IntArrayList> cluster = new IntObjectHashMap<>();
        for (int node = 0, l = Math.toIntExact(labels.size()); node < l; node++) {
            int key = Math.toIntExact(labels.get(node));
            IntArrayList ids = cluster.get(key);
            if (ids == null) {
                ids = new IntArrayList();
                cluster.put(key, ids);
            }
            ids.add(node);
        }

        return cluster;
    }

    static Stream<Arguments> expectedMemoryEstimation() {
        return Stream.of(
            Arguments.of(1, 800_480L, 4_994_656L),
            Arguments.of(4, 801_560L, 17_578_264L),
            Arguments.of(42, 815_240L, 176_970_632L)
        );
    }

    @ParameterizedTest
    @MethodSource("org.neo4j.graphalgo.labelpropagation.LabelPropagationTest#expectedMemoryEstimation")
    void shouldComputeMemoryEstimation(int concurrency, long expectedMinBytes, long expectedMaxBytes) {
        assertMemoryEstimation(
            () -> new LabelPropagationFactory<>().memoryEstimation(DEFAULT_CONFIG),
            100_000L,
            concurrency,
            expectedMinBytes,
            expectedMaxBytes
        );
    }

    @Test
    void shouldBoundMemEstimationToMaxSupportedDegree() {
        var labelPropagationFactory = new LabelPropagationFactory<>();
        GraphDimensions largeDimensions = ImmutableGraphDimensions.builder()
            .nodeCount((long) Integer.MAX_VALUE + (long) Integer.MAX_VALUE)
            .build();

        // test for no failure and no overflow
        assertTrue(0 < labelPropagationFactory
            .memoryEstimation(ImmutableLabelPropagationStreamConfig.builder().build())
            .estimate(largeDimensions, 1)
            .memoryUsage().max);
    }

    @Test
    void shouldLogProgress() {
        var testLogger = new TestProgressLogger(
            graph.relationshipCount(),
            "LabelPropagation",
            DEFAULT_CONFIG.concurrency()
        );

        var lp = new LabelPropagation(
            graph,
            DEFAULT_CONFIG,
            Pools.DEFAULT,
            testLogger,
            AllocationTracker.empty()
        );

        lp.compute();

        List<AtomicLong> progresses = testLogger.getProgresses();

        // Should log progress for every iteration + init step
        assertEquals(lp.ranIterations() + 1, progresses.size());
        progresses.forEach(progress -> assertTrue(progress.get() <= graph.relationshipCount()));

        assertTrue(testLogger.containsMessage(TestLog.INFO, ":: Start"));
        LongStream.range(1, lp.ranIterations() + 1).forEach(iteration -> {
            assertTrue(testLogger.containsMessage(TestLog.INFO, formatWithLocale("Iteration %d :: Start", iteration)));
            assertTrue(testLogger.containsMessage(TestLog.INFO, formatWithLocale("Iteration %d :: Start", iteration)));
        });
        assertTrue(testLogger.containsMessage(TestLog.INFO, ":: Finished"));
    }
}
