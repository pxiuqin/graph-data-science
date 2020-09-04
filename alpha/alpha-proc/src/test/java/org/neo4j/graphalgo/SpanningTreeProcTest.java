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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.configuration.SettingImpl;
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.core.Settings;
import org.neo4j.graphalgo.spanningtree.SpanningTreeProc;
import org.neo4j.graphdb.config.Setting;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.extension.ExtensionCallback;

import java.io.File;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 *
 *         a                a
 *     1 /   \ 2          /  \
 *      /     \          /    \
 *     b --3-- c        b      c
 *     |       |   =>   |      |
 *     4       5        |      |
 *     |       |        |      |
 *     d --6-- e        d      e
 */
public class SpanningTreeProcTest extends BaseProcTest {

    @BeforeEach
    void setup() throws Exception {
        String cypher = "CREATE(a:Node {start: true}) " +
                        "CREATE(b:Node) " +
                        "CREATE(c:Node) " +
                        "CREATE(d:Node) " +
                        "CREATE(e:Node) " +
                        "CREATE(z:Node) " +
                        "CREATE (a)-[:TYPE {cost:1.0}]->(b) " +
                        "CREATE (a)-[:TYPE {cost:2.0}]->(c) " +
                        "CREATE (b)-[:TYPE {cost:3.0}]->(c) " +
                        "CREATE (b)-[:TYPE {cost:4.0}]->(d) " +
                        "CREATE (c)-[:TYPE {cost:5.0}]->(e) " +
                        "CREATE (d)-[:TYPE {cost:6.0}]->(e)";
        runQuery(cypher);
        registerProcedures(SpanningTreeProc.class, GraphCreateProc.class);
    }

    @Override
    @ExtensionCallback
    protected void configuration(TestDatabaseManagementServiceBuilder builder) {
        super.configuration(builder);
        ClassLoader classLoader = SpanningTreeProcTest.class.getClassLoader();
        String root = new File(classLoader.getResource("transport-nodes.csv").getFile()).getParent();
        Setting<Path> setting = Settings.loadCsvFileUrlRoot();
        Path fileRoot = (((SettingImpl<Path>) setting)).parse(root);
        builder.setConfig(setting, fileRoot);
    }

    private long getStartNodeId() {
        return runQuery(
            "MATCH (n) WHERE n.start = true RETURN id(n) AS id",
            result -> result.<Long>columnAs("id").next()
        );
    }

    @Test
    void github8_testOutOfBounds() {
        String importQuery = "LOAD CSV WITH HEADERS FROM 'file:///transport-nodes.csv' AS row\n" +
                             "MERGE (place:Place {id:row.id})";
        String importRelsQuery =
                             "// Import relationships\n" +
                             "LOAD CSV WITH HEADERS FROM 'file:///transport-relationships.csv' AS row\n" +
                             "MATCH (origin:Place {id: row.src})\n" +
                             "MATCH (destination:Place {id: row.dst})\n" +
                             "MERGE (origin)-[:EROAD {distance: toInteger(row.cost)}]->(destination);";
        String insert1024NodesQuery = "";
        for (int i = 0; i < 1024; i++) {
            insert1024NodesQuery = insert1024NodesQuery + "CREATE(x" + i + ":Node) ";
        }
        String createQuery = "CALL gds.graph.create('spanningtree_example', 'Place', {" +
                             "  EROAD: {" +
                             "      type: 'EROAD'," +
                             "      orientation: 'Undirected'," +
                             "      properties: 'distance'" +
                             "  }" +
                             "})";
        String algoQuery = "MATCH (n:Place {id:\"Amsterdam\"})\n" +
                           "CALL gds.alpha.spanningTree.minimum.write('spanningtree_example', {" +
                           "    weightWriteProperty:'cost'," +
                           "    startNodeId: id(n)," +
                           "    writeProperty:'MNIST'," +
                           "    relationshipWeightProperty:'distance'" +
                           "})\n" +
                           "YIELD createMillis, computeMillis, writeMillis, effectiveNodeCount\n" +
                           "RETURN createMillis, computeMillis, writeMillis, effectiveNodeCount";
        runQuery(insert1024NodesQuery);
        runQuery(importQuery);
        runQuery(importRelsQuery);
        runQuery(createQuery);
        runQuery(algoQuery);
    }

    @Test
    void testMinimum() {
        String query = GdsCypher.call()
            .withNodeLabel("Node")
            .withRelationshipType("TYPE", Orientation.UNDIRECTED)
            .withRelationshipProperty("cost", DefaultValue.of(1.0D))
            .algo("gds.alpha.spanningTree")
            .writeMode()
            .addParameter("startNodeId", getStartNodeId())
            .addParameter("relationshipWeightProperty", "cost")
            .addParameter("weightWriteProperty", "cost")
            .yields("createMillis", "computeMillis", "writeMillis", "effectiveNodeCount");

        runQueryWithRowConsumer(
            query,
            res -> {
                assertNotEquals(-1L, res.getNumber("writeMillis").longValue());
                assertEquals(5, res.getNumber("effectiveNodeCount").intValue());
            }
        );

        final long relCount = runQuery(
            "MATCH (a)-[:MST]->(b) RETURN id(a) as a, id(b) as b",
            result -> result.stream().count()
        );

        assertEquals(relCount, 4);
    }

    @Test
    void testMaximum() {
        String query = GdsCypher.call()
            .withNodeLabel("Node")
            .withRelationshipType("TYPE", Orientation.UNDIRECTED)
            .withRelationshipProperty("cost", DefaultValue.of(1.0D))
            .algo("gds.alpha.spanningTree.maximum")
            .writeMode()
            .addParameter("startNodeId", getStartNodeId())
            .addParameter("writeProperty", "MAX")
            .addParameter("relationshipWeightProperty", "cost")
            .addParameter("weightWriteProperty", "cost")
            .yields("createMillis", "computeMillis", "writeMillis", "effectiveNodeCount");

        runQueryWithRowConsumer(
            query,
            res -> {
                assertNotEquals(-1L, res.getNumber("writeMillis").longValue());
                assertEquals(5, res.getNumber("effectiveNodeCount").intValue());
            }
        );

        long relCount = runQuery(
            "MATCH (a)-[:MAX]->(b) RETURN id(a) as a, id(b) as b",
            result -> result.stream().count()
        );

        assertEquals(relCount, 4);
    }

    @Test
    void failOnInvalidStartNode() {
        String query = GdsCypher.call()
            .loadEverything()
            .algo("gds.alpha.spanningTree.maximum")
            .writeMode()
            .addParameter("weightWriteProperty", "cost")
            .addParameter("startNodeId", 42)
            .yields();

        assertError(query, "startNode with id 42 was not loaded");
    }
}
