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
package org.neo4j.graphalgo.catalog;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.test.TestProc;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Map;

import static java.util.Collections.emptyMap;
import static java.util.Collections.singletonList;
import static java.util.Collections.singletonMap;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.compat.MapUtil.map;

class GraphCreateProcEstimateTest extends BaseProcTest {

    private static final String DB_CYPHER_ESTIMATE =
        "CREATE" +
        "  (a:A {id: 0, partition: 42})" +
        ", (b:B {id: 1, partition: 42})" +

        ", (a)-[:X { weight: 1.0 }]->(:A {id: 2,  weight: 1.0, partition: 1})" +
        ", (a)-[:X { weight: 1.0 }]->(:A {id: 3,  weight: 2.0, partition: 1})" +
        ", (a)-[:X { weight: 1.0 }]->(:A {id: 4,  weight: 1.0, partition: 1})" +
        ", (a)-[:Y { weight: 1.0 }]->(:A {id: 5,  weight: 1.0, partition: 1})" +
        ", (a)-[:Z { weight: 1.0 }]->(:A {id: 6,  weight: 8.0, partition: 2})" +

        ", (b)-[:X { weight: 42.0 }]->(:B {id: 7,  weight: 1.0, partition: 1})" +
        ", (b)-[:X { weight: 42.0 }]->(:B {id: 8,  weight: 2.0, partition: 1})" +
        ", (b)-[:X { weight: 42.0 }]->(:B {id: 9,  weight: 1.0, partition: 1})" +
        ", (b)-[:Y { weight: 42.0 }]->(:B {id: 10, weight: 1.0, partition: 1})" +
        ", (b)-[:Z { weight: 42.0 }]->(:B {id: 11, weight: 8.0, partition: 2})";

    @BeforeEach
    void setup() throws Exception {
        registerProcedures(GraphCreateProc.class, TestProc.class);
        runQuery(DB_CYPHER_ESTIMATE);
    }

    @AfterEach
    void tearDown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    @Test
    void estimateHeapPercentageForNativeProjection() {
        Map<String, Object> relProjection = map(
            "B",
            map("type", "REL")
        );
        String query = "CALL gds.graph.create.estimate('*', $relProjection)";
        double expectedPercentage = BigDecimal.valueOf(303504)
            .divide(BigDecimal.valueOf(Runtime.getRuntime().maxMemory()), 1, RoundingMode.UP)
            .doubleValue();

        runQueryWithRowConsumer(query, map("relProjection", relProjection),
            row -> {
                assertEquals(303560, row.getNumber("bytesMax").longValue());
                assertEquals(303560, row.getNumber("bytesMin").longValue());
                assertEquals(expectedPercentage, row.getNumber("heapPercentageMin").doubleValue());
                assertEquals(expectedPercentage, row.getNumber("heapPercentageMax").doubleValue());
            }
        );
    }

    @Test
    void virtualEstimateHeapPercentage() {
        Map<String, Object> relProjection = map(
            "B",
            map("type", "REL")
        );
        String query = "CALL gds.graph.create.estimate('*', $relProjection, {nodeCount: 1000000})";

        double expectedPercentage = BigDecimal.valueOf(30190200L)
            .divide(BigDecimal.valueOf(Runtime.getRuntime().maxMemory()), 1, RoundingMode.UP)
            .doubleValue();

        runQueryWithRowConsumer(query, map("relProjection", relProjection),
            row -> {
                assertEquals(30315248, row.getNumber("bytesMin").longValue());
                assertEquals(30315248, row.getNumber("bytesMax").longValue());
                assertEquals(expectedPercentage, row.getNumber("heapPercentageMin").doubleValue());
                assertEquals(expectedPercentage, row.getNumber("heapPercentageMax").doubleValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForNativeProjectionWithProperties() {
        Map<String, Object> relProjection = map(
            "B",
            map("type", "REL", "properties", "weight")
        );
        String query = "CALL gds.graph.create.estimate('*', $relProjection)";

        runQueryWithRowConsumer(query, map("relProjection", relProjection),
            row -> {
                assertEquals(573992, row.getNumber("bytesMin").longValue());
                assertEquals(573992, row.getNumber("bytesMax").longValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForCypherProjection() {
        String nodeQuery = "MATCH (n) RETURN id(n) AS id";
        String relationshipQuery = "MATCH (n)-[:REL]->(m) RETURN id(n) AS source, id(m) AS target";
        String query = "CALL gds.graph.create.cypher.estimate($nodeQuery, $relationshipQuery)";
        runQueryWithRowConsumer(
            query,
            map("nodeQuery", nodeQuery, "relationshipQuery", relationshipQuery),
            row -> {
                assertEquals(303512, row.getNumber("bytesMin").longValue());
                assertEquals(303512, row.getNumber("bytesMax").longValue());
            }
        );
    }

    @Test
    @Disabled("Disabled until we support relationshipProperties or it is removed")
    void computeMemoryEstimationForCypherProjectionWithProperties() {
        String nodeQuery = "MATCH (n) RETURN id(n) AS id";
        String relationshipQuery = "MATCH (n)-[r:REL]->(m) RETURN id(n) AS source, id(m) AS target, r.weight AS weight";
        String query = "CALL gds.graph.create.cypher.estimate($nodeQuery, $relationshipQuery)";
        runQueryWithRowConsumer(
            query,
            map("nodeQuery", nodeQuery, "relationshipQuery", relationshipQuery),
            row -> {
                assertEquals(573944, row.getNumber("bytesMin").longValue());
                assertEquals(573944, row.getNumber("bytesMax").longValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForVirtualGraph() {
        String query = "CALL gds.graph.create.estimate('*', '*', {nodeCount: 42, relationshipCount: 1337})";
        runQueryWithRowConsumer(query,
            row -> {
                assertEquals(303800, row.getNumber("bytesMin").longValue());
                assertEquals(303800, row.getNumber("bytesMax").longValue());
                assertEquals(42, row.getNumber("nodeCount").longValue());
                assertEquals(1337, row.getNumber("relationshipCount").longValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForVirtualGraphNonEmptyGraph() {
        String query = "CALL gds.graph.create.estimate('*', '*', {nodeCount: 42, relationshipCount: 1337})";
        runQueryWithRowConsumer(query,
            row -> {
                assertEquals(303800, row.getNumber("bytesMin").longValue());
                assertEquals(303800, row.getNumber("bytesMax").longValue());
                assertEquals(42, row.getNumber("nodeCount").longValue());
                assertEquals(1337, row.getNumber("relationshipCount").longValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForVirtualGraphWithProperties() throws Exception {
        String query = "CALL gds.graph.create.estimate('*', {`FOO`: {type: '*', properties: 'weight'}}, {nodeCount: 42, relationshipCount: 1337})";
        runQueryWithRowConsumer(query,
            row -> {
                assertEquals(574232, row.getNumber("bytesMin").longValue());
                assertEquals(574232, row.getNumber("bytesMax").longValue());
            }
        );
    }

    @Test
    void computeMemoryEstimationForVirtualGraphWithLargeValues() {
        String query = "CALL gds.graph.create.estimate('*', '*', {nodeCount: 5000000000, relationshipCount: 20000000000})";
        runQueryWithRowConsumer(query,
            row -> {
                assertEquals(171461586840L, row.getNumber("bytesMin").longValue());
                assertEquals(231466207488L, row.getNumber("bytesMax").longValue());
                assertEquals(5000000000L, row.getNumber("nodeCount").longValue());
                assertEquals(20000000000L, row.getNumber("relationshipCount").longValue());
            }
        );
    }

    @Test
    void multiUseLoadedGraphWithMultipleRelationships() {
        String graphName = "foo";

        String query = GdsCypher.call()
            .withAnyLabel()
            .withRelationshipType("X")
            .withRelationshipType("Y")
            .graphCreate(graphName)
            .yields("nodeCount", "relationshipCount", "graphName");

        runQueryWithRowConsumer(query, map(), resultRow -> {
                assertEquals(12L, resultRow.getNumber("nodeCount"));
                assertEquals(8L, resultRow.getNumber("relationshipCount"));
                assertEquals(graphName, resultRow.getString("graphName"));
            }
        );

        String algoQuery = "CALL gds.testProc.test('" + graphName + "', {writeProperty: 'p', relationshipTypes: $relType})";

        runQueryWithRowConsumer(algoQuery, singletonMap("relType", Arrays.asList("X", "Y")), resultRow ->
            assertEquals(8L, resultRow.getNumber("relationshipCount"))
        );

        runQueryWithRowConsumer(algoQuery, singletonMap("relType", singletonList("X")), resultRow ->
            assertEquals(6L, resultRow.getNumber("relationshipCount"))
        );

        runQueryWithRowConsumer(algoQuery, singletonMap("relType", singletonList("Y")), resultRow ->
            assertEquals(2L, resultRow.getNumber("relationshipCount"))
        );
    }

    @Test
    void silentlyDropRelsWithUnloadedNodesForCypherCreation() {
        String query = "CALL gds.graph.create.cypher(" +
                       "'g', " +
                       "'MATCH (n:A) Return id(n) as id', " +
                       "'MATCH (n)-[]->(m) RETURN id(n) AS source, id(m) AS target'," +
                       "{validateRelationships: false})" +
                       "YIELD nodeCount, relationshipCount";

        runQueryWithRowConsumer(query, resultRow -> {
            assertEquals(resultRow.getNumber("nodeCount"), 6L);
            assertEquals(resultRow.getNumber("relationshipCount"), 5L);
        });
    }

    @Test
    void silentlyDropRelsWithUnloadedNodes() {
        String query = GdsCypher.call()
            .withNodeLabel("A")
            .withAnyRelationshipType()
            .graphCreate("'g'")
            .yields("nodeCount", "relationshipCount");

        runQueryWithRowConsumer(query, resultRow -> {
            assertEquals(resultRow.getNumber("nodeCount"), 6L);
            assertEquals(resultRow.getNumber("relationshipCount"), 5L);
        });
    }

    // Failure cases

    @Test
    void failCypherCreationWitIncompleteNodeQuery() {
        String query = "CALL gds.graph.create.cypher(" +
                       "'g', " +
                       "'MATCH (n:A) Return id(n) as id', " +
                       "'MATCH (n)-[]->(m) RETURN id(n) AS source, id(m) AS target')" +
                       "YIELD nodeCount, relationshipCount";

        assertError(query, emptyMap(), "Failed to load a relationship because its source-node with id 1 is not part of the node query or projection.");
    }

    @Test
    void failCreationWitIncompleteNodeQuery() {
        String query = GdsCypher.call()
            .withNodeLabel("A")
            .withAnyRelationshipType()
            .graphCreate("'g'")
            .addParameter("validateRelationships", true)
            .yields("nodeCount");

        assertError(query, emptyMap(), "Failed to load a relationship because its source-node with id 1 is not part of the node query or projection.");
    }
}
