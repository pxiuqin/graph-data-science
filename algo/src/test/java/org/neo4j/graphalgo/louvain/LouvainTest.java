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
package org.neo4j.graphalgo.louvain;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.TestProgressLogger;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.beta.generator.RandomGraphGenerator;
import org.neo4j.graphalgo.beta.generator.RelationshipDistribution;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.huge.HugeGraph;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphalgo.core.utils.mem.MemoryTree;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeLongArray;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.IdFunction;
import org.neo4j.graphalgo.extension.Inject;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.params.provider.Arguments.arguments;
import static org.neo4j.graphalgo.CommunityHelper.assertCommunities;
import static org.neo4j.graphalgo.CommunityHelper.assertCommunitiesWithLabels;
import static org.neo4j.graphalgo.TestLog.INFO;
import static org.neo4j.graphalgo.TestSupport.assertMemoryEstimation;
import static org.neo4j.graphalgo.TestSupport.ids;
import static org.neo4j.graphalgo.core.ProcedureConstants.TOLERANCE_DEFAULT;
import static org.neo4j.graphalgo.graphbuilder.TransactionTerminationTestUtils.assertTerminates;

@GdlExtension
class LouvainTest {

    static ImmutableLouvainStreamConfig.Builder defaultConfigBuilder() {
        return ImmutableLouvainStreamConfig.builder()
            .maxLevels(10)
            .maxIterations(10)
            .tolerance(TOLERANCE_DEFAULT)
            .includeIntermediateCommunities(true)
            .concurrency(1);
    }

    @GdlGraph
    private static final String DB_CYPHER =
        "CREATE" +
        "  (a:Node {seed: 1})" +        // 0
        ", (b:Node {seed: 1})" +        // 1
        ", (c:Node {seed: 1})" +        // 2
        ", (d:Node {seed: 1})" +        // 3
        ", (e:Node {seed: 1})" +        // 4
        ", (f:Node {seed: 1})" +        // 5
        ", (g:Node {seed: 2})" +        // 6
        ", (h:Node {seed: 2})" +        // 7
        ", (i:Node {seed: 2})" +        // 8
        ", (j:Node {seed: 42})" +       // 9
        ", (k:Node {seed: 42})" +       // 10
        ", (l:Node {seed: 42})" +       // 11
        ", (m:Node {seed: 42})" +       // 12
        ", (n:Node {seed: 42})" +       // 13
        ", (x:Node {seed: 1})" +        // 14
        ", (u:Some)" +
        ", (v:Other)" +
        ", (w:Label)" +

        ", (a)-[:TYPE_OUT {weight: 1.0}]->(b)" +
        ", (a)-[:TYPE_OUT {weight: 1.0}]->(d)" +
        ", (a)-[:TYPE_OUT {weight: 1.0}]->(f)" +
        ", (b)-[:TYPE_OUT {weight: 1.0}]->(d)" +
        ", (b)-[:TYPE_OUT {weight: 1.0}]->(x)" +
        ", (b)-[:TYPE_OUT {weight: 1.0}]->(g)" +
        ", (b)-[:TYPE_OUT {weight: 1.0}]->(e)" +
        ", (c)-[:TYPE_OUT {weight: 1.0}]->(x)" +
        ", (c)-[:TYPE_OUT {weight: 1.0}]->(f)" +
        ", (d)-[:TYPE_OUT {weight: 1.0}]->(k)" +
        ", (e)-[:TYPE_OUT {weight: 1.0}]->(x)" +
        ", (e)-[:TYPE_OUT {weight: 0.01}]->(f)" +
        ", (e)-[:TYPE_OUT {weight: 1.0}]->(h)" +
        ", (f)-[:TYPE_OUT {weight: 1.0}]->(g)" +
        ", (g)-[:TYPE_OUT {weight: 1.0}]->(h)" +
        ", (h)-[:TYPE_OUT {weight: 1.0}]->(i)" +
        ", (h)-[:TYPE_OUT {weight: 1.0}]->(j)" +
        ", (i)-[:TYPE_OUT {weight: 1.0}]->(k)" +
        ", (j)-[:TYPE_OUT {weight: 1.0}]->(k)" +
        ", (j)-[:TYPE_OUT {weight: 1.0}]->(m)" +
        ", (j)-[:TYPE_OUT {weight: 1.0}]->(n)" +
        ", (k)-[:TYPE_OUT {weight: 1.0}]->(m)" +
        ", (k)-[:TYPE_OUT {weight: 1.0}]->(l)" +
        ", (l)-[:TYPE_OUT {weight: 1.0}]->(n)" +
        ", (m)-[:TYPE_OUT {weight: 1.0}]->(n)" +

        ", (a)<-[:TYPE_IN {weight: 1.0}]-(b)" +
        ", (a)<-[:TYPE_IN {weight: 1.0}]-(d)" +
        ", (a)<-[:TYPE_IN {weight: 1.0}]-(f)" +
        ", (b)<-[:TYPE_IN {weight: 1.0}]-(d)" +
        ", (b)<-[:TYPE_IN {weight: 1.0}]-(x)" +
        ", (b)<-[:TYPE_IN {weight: 1.0}]-(g)" +
        ", (b)<-[:TYPE_IN {weight: 1.0}]-(e)" +
        ", (c)<-[:TYPE_IN {weight: 1.0}]-(x)" +
        ", (c)<-[:TYPE_IN {weight: 1.0}]-(f)" +
        ", (d)<-[:TYPE_IN {weight: 1.0}]-(k)" +
        ", (e)<-[:TYPE_IN {weight: 1.0}]-(x)" +
        ", (e)<-[:TYPE_IN {weight: 0.01}]-(f)" +
        ", (e)<-[:TYPE_IN {weight: 1.0}]-(h)" +
        ", (f)<-[:TYPE_IN {weight: 1.0}]-(g)" +
        ", (g)<-[:TYPE_IN {weight: 1.0}]-(h)" +
        ", (h)<-[:TYPE_IN {weight: 1.0}]-(i)" +
        ", (h)<-[:TYPE_IN {weight: 1.0}]-(j)" +
        ", (i)<-[:TYPE_IN {weight: 1.0}]-(k)" +
        ", (j)<-[:TYPE_IN {weight: 1.0}]-(k)" +
        ", (j)<-[:TYPE_IN {weight: 1.0}]-(m)" +
        ", (j)<-[:TYPE_IN {weight: 1.0}]-(n)" +
        ", (k)<-[:TYPE_IN {weight: 1.0}]-(m)" +
        ", (k)<-[:TYPE_IN {weight: 1.0}]-(l)" +
        ", (l)<-[:TYPE_IN {weight: 1.0}]-(n)" +
        ", (m)<-[:TYPE_IN {weight: 1.0}]-(n)";

    @Inject
    private GraphStore graphStore;

    @Inject
    private IdFunction idFunction;

    @Test
    void testUnweighted() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.empty()
        );

        Louvain algorithm = new Louvain(
            graph,
            defaultConfigBuilder().build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.EMPTY
        ).withTerminationFlag(TerminationFlag.RUNNING_TRUE);

        algorithm.compute();

        final HugeLongArray[] dendrogram = algorithm.dendrograms();
        final double[] modularities = algorithm.modularities();

        assertCommunities(
            dendrogram[0],
            ids(idFunction, "a", "b", "d"),
            ids(idFunction, "c", "e", "f", "x"),
            ids(idFunction, "g", "h", "i"),
            ids(idFunction, "j", "k", "l", "m", "n")
        );

        assertCommunities(
            dendrogram[1],
            ids(idFunction, "a", "b", "c", "d", "e", "f", "x"),
            ids(idFunction, "g", "h", "i"),
            ids(idFunction, "j", "k", "l", "m", "n")
        );

        assertEquals(2, algorithm.levels());
        assertEquals(0.38, modularities[modularities.length - 1], 0.01);
    }

    @Test
    void testWeighted() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.of("weight")
        );

        Louvain algorithm = new Louvain(
            graph,
            defaultConfigBuilder().build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.EMPTY
        ).withTerminationFlag(TerminationFlag.RUNNING_TRUE);

        algorithm.compute();

        final HugeLongArray[] dendrogram = algorithm.dendrograms();
        final double[] modularities = algorithm.modularities();

        assertCommunities(
            dendrogram[0],
            ids(idFunction, "a", "b", "d"),
            ids(idFunction, "c", "e", "x"),
            ids(idFunction, "f", "g"),
            ids(idFunction, "h", "i"),
            ids(idFunction, "j", "k", "l", "m", "n")
        );

        assertCommunities(
            dendrogram[1],
            ids(idFunction, "a", "b", "c", "d", "e", "f", "g", "x"),
            ids(idFunction, "h", "i", "j", "k", "l", "m", "n")
        );

        assertEquals(2, algorithm.levels());
        assertEquals(0.37, modularities[modularities.length - 1], 0.01);
    }

    @Test
    void testSeeded() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.of("weight")
        );

        Louvain algorithm = new Louvain(
            graph,
            defaultConfigBuilder().seedProperty("seed").build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.EMPTY
        ).withTerminationFlag(TerminationFlag.RUNNING_TRUE);

        algorithm.compute();

        final HugeLongArray[] dendrogram = algorithm.dendrograms();
        final double[] modularities = algorithm.modularities();

        var expectedCommunitiesWithLabels = Map.of(
            1L, ids(idFunction, "a", "b", "c", "d", "e", "f", "x"),
            2L, ids(idFunction, "g", "h", "i"),
            42L, ids(idFunction, "j", "k", "l", "m", "n")
        );

        assertCommunitiesWithLabels(
            dendrogram[0],
            expectedCommunitiesWithLabels
        );

        assertEquals(1, algorithm.levels());
        assertEquals(0.38, modularities[modularities.length - 1], 0.01);
    }

    @Test
    void testTolerance() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.empty()
        );

        Louvain algorithm = new Louvain(
            graph,
            ImmutableLouvainStreamConfig.builder()
                .maxLevels(10)
                .maxIterations(10)
                .tolerance(2.0)
                .includeIntermediateCommunities(false)
                .concurrency(1)
                .build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.EMPTY
        ).withTerminationFlag(TerminationFlag.RUNNING_TRUE);

        algorithm.compute();

        assertEquals(1, algorithm.levels());
    }

    @Test
    void testMaxLevels() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.empty()
        );

        Louvain algorithm = new Louvain(
            graph,
            ImmutableLouvainStreamConfig.builder()
                .maxLevels(1)
                .maxIterations(10)
                .tolerance(TOLERANCE_DEFAULT)
                .includeIntermediateCommunities(false)
                .concurrency(1)
                .build(),
            Pools.DEFAULT,
            ProgressLogger.NULL_LOGGER,
            AllocationTracker.EMPTY
        ).withTerminationFlag(TerminationFlag.RUNNING_TRUE);

        algorithm.compute();

        assertEquals(1, algorithm.levels());
    }

    static Stream<Arguments> memoryEstimationTuples() {
        return Stream.of(
            arguments(1, 1, 6414145, 23941600),
            arguments(1, 10, 6414145, 31141960),
            arguments(4, 1, 6417433, 29745976),
            arguments(4, 10, 6417433, 36946336),
            arguments(42, 1, 6459081, 105719464),
            arguments(42, 10, 6459081, 112919824)
        );
    }

    @ParameterizedTest
    @MethodSource("memoryEstimationTuples")
    void testMemoryEstimation(int concurrency, int levels, long expectedMinBytes, long expectedMaxBytes) {
        GraphDimensions dimensions = ImmutableGraphDimensions.builder()
            .nodeCount(100_000L)
            .maxRelCount(500_000L)
            .build();

        LouvainStreamConfig config = ImmutableLouvainStreamConfig.builder()
            .maxLevels(levels)
            .maxIterations(10)
            .tolerance(TOLERANCE_DEFAULT)
            .includeIntermediateCommunities(false)
            .concurrency(1)
            .build();

        assertMemoryEstimation(
            () -> new LouvainFactory<>().memoryEstimation(config),
            dimensions,
            concurrency,
            expectedMinBytes,
            expectedMaxBytes
        );
    }

    @Test
    void testMemoryEstimationUsesOnlyOnePropertyForEachEntity() {
        ImmutableGraphDimensions.Builder dimensionsBuilder = ImmutableGraphDimensions.builder()
            .nodeCount(100_000L)
            .maxRelCount(500_000L);

        GraphDimensions dimensionsWithoutProperties = dimensionsBuilder.build();
        GraphDimensions dimensionsWithOneProperty = dimensionsBuilder
            .putRelationshipPropertyToken("foo", 1)
            .build();
        GraphDimensions dimensionsWithTwoProperties = dimensionsBuilder
            .putRelationshipPropertyToken("foo", 1)
            .putRelationshipPropertyToken("bar", 1)
            .build();

        LouvainStreamConfig config = ImmutableLouvainStreamConfig.builder()
            .maxLevels(1)
            .maxIterations(10)
            .tolerance(TOLERANCE_DEFAULT)
            .includeIntermediateCommunities(false)
            .concurrency(1)
            .build();

        MemoryTree memoryTree = new LouvainFactory<>()
            .memoryEstimation(config)
            .estimate(dimensionsWithoutProperties, 1);
        MemoryTree memoryTreeOneProperty = new LouvainFactory<>()
            .memoryEstimation(config)
            .estimate(dimensionsWithOneProperty, 1);
        MemoryTree memoryTreeTwoProperties = new LouvainFactory<>()
            .memoryEstimation(config)
            .estimate(dimensionsWithTwoProperties, 1);

        assertEquals(memoryTree.memoryUsage(), memoryTreeOneProperty.memoryUsage());
        assertEquals(memoryTreeOneProperty.memoryUsage(), memoryTreeTwoProperties.memoryUsage());
    }

    @Test
    void testCanBeInterruptedByTxCancellation() {
        HugeGraph graph = RandomGraphGenerator.builder()
            .nodeCount(100_000)
            .averageDegree(10)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .allocationTracker(AllocationTracker.EMPTY)
            .build()
            .generate();

        assertTerminates((terminationFlag) ->
            new Louvain(
                graph,
                defaultConfigBuilder().concurrency(2).build(),
                Pools.DEFAULT,
                ProgressLogger.NULL_LOGGER,
                AllocationTracker.EMPTY
            )
                .withTerminationFlag(terminationFlag)
                .compute(), 500, 1000
        );
    }

    @Test
    void testLogging() {
        var graph = graphStore.getGraph(
            NodeLabel.listOf("Node"),
            RelationshipType.listOf("TYPE_OUT", "TYPE_IN"),
            Optional.empty()
        );

        var config = defaultConfigBuilder().build();

        var testLogger = new TestProgressLogger(0, "Louvain", config.concurrency());

        var louvain = new Louvain(
            graph,
            config,
            Pools.DEFAULT,
            testLogger,
            AllocationTracker.EMPTY
        );

        louvain.compute();

        assertTrue(testLogger.containsMessage(INFO, ":: Start"));
        assertTrue(testLogger.containsMessage(INFO, "Level 1 :: Finished"));
        assertTrue(testLogger.containsMessage(INFO, ":: Finished"));
    }
}
