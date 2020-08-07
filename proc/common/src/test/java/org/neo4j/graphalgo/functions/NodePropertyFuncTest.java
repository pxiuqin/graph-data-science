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
package org.neo4j.graphalgo.functions;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;

import java.util.Arrays;

import static org.neo4j.graphalgo.compat.MapUtil.map;

class NodePropertyFuncTest  extends BaseProcTest {

    private static final String DB_CYPHER =
        "CREATE " +
        "  (:A { prop: 42.0 })" +
        ", (:B { prop: 84.0 })";

    @BeforeEach
    void setUp() throws Exception {
        registerProcedures(GraphCreateProc.class);
        registerFunctions(NodePropertyFunc.class);
        runQuery(DB_CYPHER);
        runQuery(GdsCypher
            .call()
            .withNodeLabel("A")
            .withNodeLabel("B")
            .withNodeProperty("prop")
            .withAnyRelationshipType()
            .graphCreate("testGraph")
            .yields());
    }

    @AfterEach
    void tearDown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    @Test
    void shouldReturnNodeProperty() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('testGraph', id(n), 'prop') AS prop ORDER BY prop ASC";
        assertCypherResult(query, Arrays.asList(map("prop", 42.0), map("prop", 84.0)));
    }

    @Test
    void shouldReturnNodePropertyForLabel() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('testGraph', id(n), 'prop', 'A') AS prop ORDER BY prop ASC";
        assertCypherResult(query, Arrays.asList(map("prop", 42.0), map("prop", null)));
    }

    @Test
    void failsOnNonExistingGraph() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('noGraph', id(n), 'prop') AS prop";
        assertError(query, "Cannot find graph with name 'noGraph'.");
    }

    @Test
    void failsOnNonExistingNode() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('testGraph', 42, 'prop') AS prop";
        assertError(query, "Node id 42 does not exist.");
    }

    @Test
    void failsOnNonExistingProperty() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('testGraph', id(n), 'noProp') AS prop";
        assertError(query, "No node projection with property 'noProp' exists.");
    }

    @Test
    void failsOnNonExistingPropertyForLabel() {
        String query = "MATCH (n) RETURN gds.util.nodeProperty('testGraph', id(n), 'noProp', 'A') AS prop";
        assertError(query, "Node projection 'A' does not have property key 'noProp'. Available keys: ['prop'].");
    }
}
