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
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.nodesim.NodeSimilarityMutateProc;
import org.neo4j.graphalgo.wcc.WccMutateProc;

import static java.util.Collections.singletonList;
import static org.neo4j.graphalgo.compat.MapUtil.map;

class GraphSchemaWithMutationTest extends BaseProcTest {

    private static final String DB_CYPHER = "CREATE (:A {foo: 1})-[:REL {bar: 2}]->(:A)";

    @BeforeEach
    void setup() throws Exception {
        registerProcedures(
            GraphCreateProc.class,
            GraphListProc.class,
            WccMutateProc.class,
            NodeSimilarityMutateProc.class
        );
        runQuery(DB_CYPHER);
    }

    @AfterEach
    void tearDown() { GraphStoreCatalog.removeAllLoadedGraphs();}

    @Test
    void listWithMutatedNodeProperty() {
        String name = "name";
        runQuery(
            "CALL gds.graph.create($name, 'A', 'REL', {nodeProperties: 'foo', relationshipProperties: 'bar'})",
            map("name", name)
        );
        runQuery(
            "CALL gds.wcc.mutate($name, {mutateProperty: 'baz'})",
            map("name", name)
        );

        assertCypherResult("CALL gds.graph.list() YIELD schema", singletonList(
            map(
                "schema", map(
                    "nodes", map("A", map("foo", "Integer", "baz", "Integer")),
                    "relationships", map("REL", map("bar", "Float"))
                )
            )
        ));
    }

    @Test
    void listWithMutatedRelationshipProperty() {
        runQuery("CALL gds.graph.create('graph', 'A', 'REL', {nodeProperties: 'foo', relationshipProperties: 'bar'})");
        runQuery("CALL gds.nodeSimilarity.mutate('graph', {mutateProperty: 'faz', mutateRelationshipType: 'BOO'})");

        assertCypherResult("CALL gds.graph.list() YIELD schema",
            singletonList(
                map(
                    "schema",
                    map("nodes",
                        map("A", map("foo", "Integer")),
                        "relationships",
                        map("BOO", map("faz", "Float"),
                            "REL", map("bar", "Float"))
                    )
                )
            )
        );
    }

}
