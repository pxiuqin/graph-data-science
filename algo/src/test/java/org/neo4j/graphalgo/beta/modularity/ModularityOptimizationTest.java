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
package org.neo4j.graphalgo.beta.modularity;

import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.TestProgressLogger;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryTree;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.IdFunction;
import org.neo4j.graphalgo.extension.Inject;
import org.neo4j.graphalgo.extension.TestGraph;
import org.neo4j.internal.kernel.api.security.AuthSubject;

import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.params.provider.Arguments.arguments;
import static org.neo4j.graphalgo.CommunityHelper.assertCommunities;
import static org.neo4j.graphalgo.TestLog.INFO;
import static org.neo4j.graphalgo.TestSupport.ids;
import static org.neo4j.graphalgo.core.ProcedureConstants.TOLERANCE_DEFAULT;

@GdlExtension
class ModularityOptimizationTest {

    private static final String[][] EXPECTED_SEED_COMMUNITIES = {new String[]{"a", "b"}, new String[]{"c", "e"}, new String[]{"d", "f"}};

    @GdlGraph
    private static final String DB_CYPHER =
        "CREATE" +
        "  (a:Node {seed1:  1,  seed2: 21})" +
        ", (b:Node {seed1: -1,  seed2: -1})" +
        ", (c:Node {seed1:  2,  seed2: 42})" +
        ", (d:Node {seed1:  3,  seed2: 33})" +
        ", (e:Node {seed1:  2,  seed2: 42})" +
        ", (f:Node {seed1:  3,  seed2: 33})" +

        ", (a)-[:TYPE_OUT {weight: 0.01}]->(b)" +
        ", (a)-[:TYPE_OUT {weight: 5.0}]->(e)" +
        ", (a)-[:TYPE_OUT {weight: 5.0}]->(f)" +
        ", (b)-[:TYPE_OUT {weight: 5.0}]->(c)" +
        ", (b)-[:TYPE_OUT {weight: 5.0}]->(d)" +
        ", (c)-[:TYPE_OUT {weight: 0.01}]->(e)" +
        ", (f)-[:TYPE_OUT {weight: 0.01}]->(d)" +

        ", (a)<-[:TYPE_IN {weight: 0.01}]-(b)" +
        ", (a)<-[:TYPE_IN {weight: 5.0}]-(e)" +
        ", (a)<-[:TYPE_IN {weight: 5.0}]-(f)" +
        ", (b)<-[:TYPE_IN {weight: 5.0}]-(c)" +
        ", (b)<-[:TYPE_IN {weight: 5.0}]-(d)" +
        ", (c)<-[:TYPE_IN {weight: 0.01}]-(e)" +
        ", (f)<-[:TYPE_IN {weight: 0.01}]-(d)";

    @Inject
    private TestGraph graph;

    @Inject
    private GraphStore graphStore;

    @Inject
    private IdFunction idFunction;

    @Test
    void testUnweighted() {
        var graph = unweightedGraph();

        ModularityOptimization pmo = compute(graph, 3, null, 1, 10_000, ProgressLogger.NULL_LOGGER);

        assertEquals(0.12244, pmo.getModularity(), 0.001);
        assertCommunities(
            getCommunityIds(graph.nodeCount(), pmo),
            ids(idFunction, "a", "b", "c", "e"),
            ids(idFunction, "d", "f")
        );
        assertTrue(pmo.getIterations() <= 3);
    }

    @Test
    void testWeighted() {
        ModularityOptimization pmo = compute(graph, 3, null, 3, 2, ProgressLogger.NULL_LOGGER);

        assertEquals(0.4985, pmo.getModularity(), 0.001);
        assertCommunities(
            getCommunityIds(graph.nodeCount(), pmo),
            ids(idFunction, "a", "e", "f"),
            ids(idFunction, "b", "c", "d")
        );
        assertTrue(pmo.getIterations() <= 3);
    }

    @Test
    void testSeedingWithBiggerSeedValues() {
        var graph = unweightedGraph();

        ModularityOptimization pmo = compute(
            graph,
            10, graph.nodeProperties("seed2"),
            1,
            100,
            ProgressLogger.NULL_LOGGER
        );

        long[] actualCommunities = getCommunityIds(graph.nodeCount(), pmo);
        assertEquals(0.0816, pmo.getModularity(), 0.001);
        assertCommunities(actualCommunities, ids(idFunction, EXPECTED_SEED_COMMUNITIES));
        assertTrue(actualCommunities[0] == 43 && actualCommunities[2] == 42 && actualCommunities[3] == 33);
        assertTrue(pmo.getIterations() <= 3);
    }

    @Test
    void testSeeding() {
        var graph = unweightedGraph();

        ModularityOptimization pmo = compute(
            graph,
            10, graph.nodeProperties("seed1"),
            1,
            100,
            ProgressLogger.NULL_LOGGER
        );

        long[] actualCommunities = getCommunityIds(graph.nodeCount(), pmo);
        assertEquals(0.0816, pmo.getModularity(), 0.001);
        assertCommunities(actualCommunities, ids(idFunction, EXPECTED_SEED_COMMUNITIES));
        assertTrue(actualCommunities[0] == 4 && actualCommunities[2] == 2 || actualCommunities[3] == 3);
        assertTrue(pmo.getIterations() <= 3);
    }

    private long[] getCommunityIds(long nodeCount, ModularityOptimization pmo) {
        long[] communityIds = new long[(int) nodeCount];
        for (int i = 0; i < nodeCount; i++) {
            communityIds[i] = pmo.getCommunityId(i);
        }
        return communityIds;
    }

    @Test
    void testLogging() {
        var testLogger = new TestProgressLogger(
            graph.relationshipCount(),
            "ModularityOptimization",
            3
        );

        compute(graph, 3, null, 3, 2, testLogger);

        assertTrue(testLogger.containsMessage(INFO, ":: Start"));
        assertTrue(testLogger.containsMessage(INFO, "Initialization :: Start"));
        assertTrue(testLogger.containsMessage(INFO, "Initialization :: Finished"));
        assertTrue(testLogger.containsMessage(INFO, "Iteration 1 :: Start"));
        assertTrue(testLogger.containsMessage(INFO, "Iteration 1 :: Finished"));
        assertTrue(testLogger.containsMessage(INFO, ":: Finished"));
    }

    @Test
    void requireAtLeastOneIteration() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> compute(graph, 0, null, 3, 2, ProgressLogger.NULL_LOGGER)
        );

        assertTrue(exception.getMessage().contains("at least one iteration"));
    }

    @ParameterizedTest
    @MethodSource("memoryEstimationTuples")
    void testMemoryEstimation(int concurrency, long min, long max) {
        GraphDimensions dimensions = ImmutableGraphDimensions.builder().nodeCount(100_000L).build();

        ModularityOptimizationStreamConfig config = ImmutableModularityOptimizationStreamConfig.builder()
            .username(AuthSubject.ANONYMOUS.username())
            .graphName("")
            .build();
        MemoryTree memoryTree = new ModularityOptimizationFactory<>()
            .memoryEstimation(config)
            .estimate(dimensions, concurrency);
        assertEquals(min, memoryTree.memoryUsage().min);
        assertEquals(max, memoryTree.memoryUsage().max);
    }

    static Stream<Arguments> memoryEstimationTuples() {
        return Stream.of(
            arguments(1, 5614048, 8413080),
            arguments(4, 5617336, 14413344),
            arguments(42, 5658984, 90416688)
        );
    }

    @NotNull
    private ModularityOptimization compute(
        Graph graph,
        int maxIterations,
        NodeProperties properties,
        int concurrency,
        int minBatchSize,
        ProgressLogger testLogger
    ) {
        return new ModularityOptimization(
            graph,
            maxIterations,
            TOLERANCE_DEFAULT,
            properties,
            concurrency,
            minBatchSize,
            Pools.DEFAULT,
            testLogger,
            AllocationTracker.EMPTY
        ).compute();
    }

    private Graph unweightedGraph() {
        return graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.empty()
        );
    }
}
