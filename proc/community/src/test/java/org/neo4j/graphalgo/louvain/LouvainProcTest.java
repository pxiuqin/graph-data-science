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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.provider.Arguments;
import org.neo4j.graphalgo.AlgoBaseProcTest;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.ConsecutiveIdsConfigTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.HeapControlTest;
import org.neo4j.graphalgo.IterationsConfigTest;
import org.neo4j.graphalgo.MemoryEstimateTest;
import org.neo4j.graphalgo.NodeProjections;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.PropertyMappings;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipProjections;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.RelationshipWeightConfigTest;
import org.neo4j.graphalgo.SeedConfigTest;
import org.neo4j.graphalgo.ToleranceConfigTest;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.catalog.GraphWriteNodePropertiesProc;
import org.neo4j.graphalgo.config.ImmutableGraphCreateFromStoreConfig;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.functions.AsNodeFunc;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.params.provider.Arguments.arguments;

abstract class LouvainProcTest<CONFIG extends LouvainBaseConfig> extends BaseProcTest implements
    AlgoBaseProcTest<Louvain, CONFIG, Louvain>,
    SeedConfigTest<Louvain, CONFIG, Louvain>,
    ConsecutiveIdsConfigTest<Louvain, CONFIG, Louvain>,
    IterationsConfigTest<Louvain, CONFIG, Louvain>,
    RelationshipWeightConfigTest<Louvain, CONFIG, Louvain>,
    ToleranceConfigTest<Louvain, CONFIG, Louvain>,
    MemoryEstimateTest<Louvain, CONFIG, Louvain>,
    HeapControlTest<Louvain, CONFIG, Louvain> {

    static final List<List<Long>> RESULT = Arrays.asList(
        Arrays.asList(0L, 1L, 2L, 3L, 4L, 5L, 14L),
        Arrays.asList(6L, 7L, 8L),
        Arrays.asList(9L, 10L, 11L, 12L, 13L)
    );

    static final String LOUVAIN_GRAPH = "myGraph";

    @Override
    public String createQuery() {
        return "CREATE" +
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

               ", (a)-[:TYPE {weight: 1.0}]->(b)" +
               ", (a)-[:TYPE {weight: 1.0}]->(d)" +
               ", (a)-[:TYPE {weight: 1.0}]->(f)" +
               ", (b)-[:TYPE {weight: 1.0}]->(d)" +
               ", (b)-[:TYPE {weight: 1.0}]->(x)" +
               ", (b)-[:TYPE {weight: 1.0}]->(g)" +
               ", (b)-[:TYPE {weight: 1.0}]->(e)" +
               ", (c)-[:TYPE {weight: 1.0}]->(x)" +
               ", (c)-[:TYPE {weight: 1.0}]->(f)" +
               ", (d)-[:TYPE {weight: 1.0}]->(k)" +
               ", (e)-[:TYPE {weight: 1.0}]->(x)" +
               ", (e)-[:TYPE {weight: 0.01}]->(f)" +
               ", (e)-[:TYPE {weight: 1.0}]->(h)" +
               ", (f)-[:TYPE {weight: 1.0}]->(g)" +
               ", (g)-[:TYPE {weight: 1.0}]->(h)" +
               ", (h)-[:TYPE {weight: 1.0}]->(i)" +
               ", (h)-[:TYPE {weight: 1.0}]->(j)" +
               ", (i)-[:TYPE {weight: 1.0}]->(k)" +
               ", (j)-[:TYPE {weight: 1.0}]->(k)" +
               ", (j)-[:TYPE {weight: 1.0}]->(m)" +
               ", (j)-[:TYPE {weight: 1.0}]->(n)" +
               ", (k)-[:TYPE {weight: 1.0}]->(m)" +
               ", (k)-[:TYPE {weight: 1.0}]->(l)" +
               ", (l)-[:TYPE {weight: 1.0}]->(n)" +
               ", (m)-[:TYPE {weight: 1.0}]->(n)";
    }

    @Override
    public GraphDatabaseAPI graphDb() {
        return db;
    }

    @BeforeEach
    void setupGraph() throws Exception {
        registerProcedures(
            LouvainStreamProc.class,
            LouvainWriteProc.class,
            LouvainStatsProc.class,
            LouvainMutateProc.class,
            GraphCreateProc.class,
            GraphWriteNodePropertiesProc.class
        );
        registerFunctions(AsNodeFunc.class);

        runQuery(createQuery());
        graphCreateQueries().forEach(this::runQuery);
    }

    List<String> graphCreateQueries() {
        return singletonList(
            GdsCypher.call()
                .withNodeLabel("Node")
                .withNodeProperty("seed")
                .withRelationshipType(
                    "TYPE",
                    RelationshipProjection.of(
                        "TYPE",
                        Orientation.UNDIRECTED,
                        Aggregation.DEFAULT
                    )
                )
                .graphCreate(LOUVAIN_GRAPH)
                .yields()
        );
    }

    @AfterEach
    void clearCommunities() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    static Stream<Arguments> graphVariations() {
        return Stream.of(
            arguments(
                GdsCypher.call().explicitCreation("myGraph"),
                "explicit graph"
            ),
            arguments(
                GdsCypher.call().implicitCreation(ImmutableGraphCreateFromStoreConfig
                    .builder()
                    .graphName("")
                    .nodeProjections(NodeProjections.fromString("Node"))
                    .nodeProperties(PropertyMappings.fromObject("seed"))
                    .relationshipProjections(RelationshipProjections.builder()
                        .putProjection(
                            RelationshipType.of("TYPE"),
                            RelationshipProjection.builder()
                                .type("TYPE")
                                .orientation(Orientation.UNDIRECTED)
                                .build()
                        )
                        .build()
                    )
                    .build()
                ),
                "implicit graph"
            )
        );
    }

    @Override
    public void assertResultEquals(Louvain result1, Louvain result2) {
        assertEquals(result1.levels(), result2.levels());
        assertEquals(result1.modularities()[result1.levels() - 1], result2.modularities()[result2.levels() - 1]);
        assertArrayEquals(result1.finalDendrogram().toArray(), result2.finalDendrogram().toArray());
    }
}
