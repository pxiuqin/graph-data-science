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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.provider.Arguments;
import org.neo4j.graphalgo.AlgoBaseProcTest;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.ConsecutiveIdsConfigTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.HeapControlTest;
import org.neo4j.graphalgo.IterationsConfigTest;
import org.neo4j.graphalgo.MemoryEstimateTest;
import org.neo4j.graphalgo.NodeProjections;
import org.neo4j.graphalgo.NodeWeightConfigTest;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.PropertyMappings;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipProjections;
import org.neo4j.graphalgo.RelationshipWeightConfigTest;
import org.neo4j.graphalgo.SeedConfigTest;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.catalog.GraphWriteNodePropertiesProc;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.config.ImmutableGraphCreateFromStoreConfig;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.extension.ExtensionCallback;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.params.provider.Arguments.arguments;
import static org.neo4j.graphalgo.RelationshipType.ALL_RELATIONSHIPS;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

abstract class LabelPropagationProcTest<CONFIG extends LabelPropagationBaseConfig> extends BaseProcTest implements
    AlgoBaseProcTest<LabelPropagation, CONFIG, LabelPropagation>,
    SeedConfigTest<LabelPropagation, CONFIG, LabelPropagation>,
    ConsecutiveIdsConfigTest<LabelPropagation, CONFIG, LabelPropagation>,
    IterationsConfigTest<LabelPropagation, CONFIG, LabelPropagation>,
    NodeWeightConfigTest<LabelPropagation, CONFIG, LabelPropagation>,
    RelationshipWeightConfigTest<LabelPropagation, CONFIG, LabelPropagation>,
    MemoryEstimateTest<LabelPropagation, CONFIG, LabelPropagation>,
    HeapControlTest<LabelPropagation, CONFIG, LabelPropagation>
{

    static final List<Long> RESULT = Arrays.asList(2L, 7L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L);
    static final String TEST_GRAPH_NAME = "myGraph";
    private static final String TEST_CYPHER_GRAPH_NAME = "myCypherGraph";

    private static final String nodeQuery = "MATCH (n) RETURN id(n) AS id, n.weight AS weight, n.seed AS seed";
    private static final String relQuery = "MATCH (s)-[:X]->(t) RETURN id(s) AS source, id(t) AS target";

    @Override
    public String createQuery() {
        return "CREATE" +
               "  (a:A {id: 0, seed: 42}) " +
               ", (b:B {id: 1, seed: 42}) " +

               ", (a)-[:X]->(:A {id: 2,  weight: 1.0, seed: 1}) " +
               ", (a)-[:X]->(:A {id: 3,  weight: 2.0, seed: 1}) " +
               ", (a)-[:X]->(:A {id: 4,  weight: 1.0, seed: 1}) " +
               ", (a)-[:X]->(:A {id: 5,  weight: 1.0, seed: 1}) " +
               ", (a)-[:X]->(:A {id: 6,  weight: 8.0, seed: 2}) " +

               ", (b)-[:X]->(:B {id: 7,  weight: 1.0, seed: 1}) " +
               ", (b)-[:X]->(:B {id: 8,  weight: 2.0, seed: 1}) " +
               ", (b)-[:X]->(:B {id: 9,  weight: 1.0, seed: 1}) " +
               ", (b)-[:X]->(:B {id: 10, weight: 1.0, seed: 1}) " +
               ", (b)-[:X]->(:B {id: 11, weight: 8.0, seed: 2})";
    }

    @Override
    public GraphDatabaseAPI graphDb() {
        return db;
    }

    @BeforeEach
    void setupGraph() throws Exception {
        registerProcedures(
            LabelPropagationStreamProc.class,
            LabelPropagationWriteProc.class,
            LabelPropagationStatsProc.class,
            LabelPropagationMutateProc.class,
            GraphCreateProc.class,
            GraphWriteNodePropertiesProc.class
        );
        runQuery(createQuery());
        // Create explicit graphs with both projection variants
        runQuery(graphCreateQuery(Orientation.NATURAL, TEST_GRAPH_NAME));
        runQuery(formatWithLocale(
            "CALL gds.graph.create.cypher('%s', '%s', '%s', {})",
            TEST_CYPHER_GRAPH_NAME,
            nodeQuery,
            relQuery
        ));
    }

    @Override
    @ExtensionCallback
    protected void configuration(TestDatabaseManagementServiceBuilder builder) {
        super.configuration(builder);
    }

    @AfterEach
    void clearCommunities() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    static String graphCreateQuery(Orientation orientation, String graphName) {
        return graphCreateQuery(orientation).graphCreate(graphName).yields();
    }

    static GdsCypher.QueryBuilder graphCreateQuery(Orientation orientation) {
        return GdsCypher
            .call()
            .implicitCreation(ImmutableGraphCreateFromStoreConfig
                .builder()
                .graphName("")
                .nodeProjections(NodeProjections.fromObject(MapUtil.map("A", "A", "B", "B")))
                .nodeProperties(PropertyMappings.fromObject(Arrays.asList("seed", "weight")))
                .relationshipProjections(RelationshipProjections.builder()
                    .putProjection(
                        ALL_RELATIONSHIPS,
                        RelationshipProjection.builder()
                            .type("X")
                            .orientation(orientation)
                            .build()
                    )
                    .build()
                )
                .build()
             );
    }

    static Stream<Arguments> gdsGraphVariations() {
        return Stream.of(
            arguments(
                GdsCypher.call().explicitCreation(TEST_GRAPH_NAME),
                "explicit graph"
            ),
            arguments(
                graphCreateQuery(Orientation.NATURAL),
                "implicit graph"
            )
        );
    }

    @Override
    public void assertResultEquals(LabelPropagation result1, LabelPropagation result2) {
        assertArrayEquals(result1.labels().toArray(), result2.labels().toArray());
        assertEquals(result1.didConverge(), result2.didConverge());
        assertEquals(result1.ranIterations(), result2.ranIterations());
    }

    @Override
    @Disabled("The algorithm is using Neo4j Node IDs to compute labels hence can't compare offset results")
    public void testNonConsecutiveIds() {}
}
