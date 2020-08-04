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
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.TestSupport;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.nodesim.NodeSimilarityMutateProc;

import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.neo4j.graphalgo.compat.MapUtil.map;

public class DeleteRelationshipsIntegrationTest extends BaseProcTest {

    private static final String TEST_GRAPH2 = "testGraph2";

    private static final String DB_CYPHER =
        "CREATE" +
        "  (a:Node { nodeId: 0 })" +
        ", (b:Node { nodeId: 1 })" +
        ", (c:Node { nodeId: 2 })" +
        ", (d:Node { nodeId: 3 })" +
        ", (e:Node { nodeId: 4 })" +
        ", (f:Node { nodeId: 5 })" +
        ", (g:Node { nodeId: 6 })" +
        ", (h:Node { nodeId: 7 })" +
        ", (i:Node { nodeId: 8 })" +
        ", (j:Node { nodeId: 9 })" +
        ", (k:Node { nodeId: 10 })" +
        ", (l:Node { nodeId: 11 })" +
        ", (a)-[:TYPE {p: 10}]->(b)" +
        ", (b)-[:TYPE]->(c)" +
        ", (c)-[:TYPE]->(d)" +
        ", (d)-[:TYPE]->(e)" +
        ", (e)-[:TYPE]->(f)" +
        ", (f)-[:TYPE]->(g)" +
        ", (h)-[:TYPE]->(i)" +
        ", (i)-[:TYPE]->(k)" +
        ", (i)-[:TYPE]->(l)" +
        ", (j)-[:TYPE]->(k)";

    @BeforeEach
    void setup() throws Exception {
        registerProcedures(GraphCreateProc.class, NodeSimilarityMutateProc.class, GraphDeleteRelationshipProc.class);
        runQuery(DB_CYPHER);

        runQuery(GdsCypher
            .call()
            .withAnyLabel()
            .withNodeProperty("nodeId")
            .withRelationshipType("TYPE")
            .graphCreate("testGraph")
            .yields());

        runQuery(GdsCypher
            .call()
            .withAnyLabel()
            .withNodeProperty("nodeId")
            .withRelationshipType("TYPE")
            .withRelationshipProperty("p", 2.0)
            .graphCreate(TEST_GRAPH2)
            .yields());
    }

    @AfterEach
    void tearDown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    @Test
    void shouldBeAbleToMutateAndDelete() {
        Graph graphBefore = GraphStoreCatalog.get(getUsername(), db.databaseId(), TEST_GRAPH2).graphStore().getUnion();

        runQuery("CALL gds.nodeSimilarity.mutate('testGraph2', {mutateRelationshipType: 'SIM', mutateProperty: 'foo'})");
        Graph graphAfterMutate = GraphStoreCatalog.get(getUsername(), db.databaseId(), TEST_GRAPH2).graphStore().getUnion();
        assertNotEquals(graphBefore.relationshipCount(), graphAfterMutate.relationshipCount());

        assertCypherResult(
            "CALL gds.graph.deleteRelationships('testGraph2', 'SIM') YIELD deletedProperties",
            singletonList(map("deletedProperties", map("foo", 2L)))
        );
        Graph graphAfterDelete = GraphStoreCatalog.get(getUsername(), db.databaseId(), TEST_GRAPH2).graphStore().getUnion();
        TestSupport.assertGraphEquals(graphBefore, graphAfterDelete);
    }

    @Test
    void shouldNotDeletePropertiesWhenNoneOnTheRelType() {
        runQuery("CALL gds.nodeSimilarity.mutate('testGraph', {mutateRelationshipType: 'SIM', mutateProperty: 'foo'})");

        assertCypherResult(
            "CALL gds.graph.deleteRelationships('testGraph', 'TYPE') YIELD deletedProperties",
            singletonList(map("deletedProperties", map()))
        );
    }
}
