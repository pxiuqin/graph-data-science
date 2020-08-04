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
package org.neo4j.graphalgo.core.huge;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.api.CSRGraph;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.Inject;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@GdlExtension
class CompositeAdjacencyListTest {

    @GdlGraph
    private static final String DB_CYPHER =
        "CREATE " +
        "  (a)" +
        ", (b)" +
        ", (a)-[:T1]->(b)" +
        ", (a)-[:T2]->(c)";

    @Inject
    Graph graph;

    @Test
    void shouldComputeCorrectDegree() {
        var adjacencyList = ((CSRGraph) graph).relationshipTopology().list();
        assertTrue(adjacencyList instanceof CompositeAdjacencyList);
        assertEquals(2, adjacencyList.degree(0));
        assertEquals(0, adjacencyList.degree(1));
    }
}
