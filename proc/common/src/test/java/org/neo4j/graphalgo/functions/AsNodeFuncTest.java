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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Result;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class AsNodeFuncTest extends BaseProcTest {

    @BeforeEach
    void setUp() throws Exception {
        registerFunctions(AsNodeFunc.class);
    }

    @Test
    void lookupNode() {
        String createNodeQuery = "CREATE (p:Person {name: 'Mark'}) RETURN p AS node";
        Node savedNode = (Node) runQuery(createNodeQuery, Result::next).get("node");

        Map<String, Object> params = MapUtil.map("nodeId", savedNode.getId());
        Map<String, Object> row = runQuery("RETURN gds.util.asNode($nodeId) AS node", params, Result::next);

        Node node = (Node) row.get("node");
        assertEquals(savedNode, node);
    }

    @Test
    void lookupNonExistentNode() {
        Map<String, Object> row = runQuery(
                "RETURN gds.util.asNode(3) AS node", Result::next);

        assertNull(row.get("node"));
    }

    @Test
    void lookupNodes() {
        String createNodeQuery = "CREATE (p1:Person {name: 'Mark'}) CREATE (p2:Person {name: 'Arya'}) RETURN p1, p2";
        Map<String, Object> savedRow = runQuery(createNodeQuery, Result::next);
        Node savedNode1 = (Node) savedRow.get("p1");
        Node savedNode2 = (Node) savedRow.get("p2");

        Map<String, Object> params = MapUtil.map("nodeIds", Arrays.asList(savedNode1.getId(), savedNode2.getId()));
        Map<String, Object> row = runQuery("RETURN gds.util.asNodes($nodeIds) AS nodes", params, Result::next);

        List<Node> nodes = (List<Node>) row.get("nodes");
        assertEquals(Arrays.asList(savedNode1, savedNode2), nodes);
    }

    @Test
    void lookupNonExistentNodes() {
        Map<String, Object> row = runQuery(
                "RETURN gds.util.asNodes([3,4,5]) AS nodes", Result::next);

        List<Node> nodes = (List<Node>) row.get("nodes");
        assertEquals(0, nodes.size());
    }

}
