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
package org.neo4j.graphalgo.triangle;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.WritePropertyConfigTest;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.isA;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

class TriangleCountWriteProcTest
    extends TriangleCountBaseProcTest<TriangleCountWriteConfig>
    implements WritePropertyConfigTest<IntersectingTriangleCount, TriangleCountWriteConfig, IntersectingTriangleCount.TriangleCountResult> {

    @Override
    public String createQuery() {
        return "CREATE " +
               "(a:A { name: 'a' })-[:T]->(b:A { name: 'b' }), " +
               "(b)-[:T]->(c:A { name: 'c' }), " +
               "(c)-[:T]->(a), " +
               "(a)-[:T]->(d:A { name: 'd' }), " +
               "(b)-[:T]->(d), " +
               "(c)-[:T]->(d), " +
               "(a)-[:T]->(e:A { name: 'e' }), " +
               "(b)-[:T]->(e) ";
    }

    @Test
    void testWrite() {
        var query = GdsCypher.call()
            .loadEverything(Orientation.UNDIRECTED)
            .algo("triangleCount")
            .writeMode()
            .addParameter("writeProperty", "triangles")
            .yields();

        assertCypherResult(query, List.of(Map.of(
            "globalTriangleCount", 5L,
            "nodeCount", 5L,
            "createMillis", greaterThan(-1L),
            "computeMillis", greaterThan(-1L),
            "configuration", isA(Map.class),
            "nodePropertiesWritten", 5L,
            "writeMillis", greaterThan(-1L)
        )));

        Map<String, Long> expectedResult = Map.of(
            "a", 4L,
            "b", 4L,
            "c", 3L,
            "d", 3L,
            "e", 1L
        );

        assertWriteResult(expectedResult, "triangles");
    }

    @Test
    void testWriteWithMaxDegree() {
        var query = GdsCypher.call()
            .loadEverything(Orientation.UNDIRECTED)
            .algo("triangleCount")
            .writeMode()
            .addParameter("writeProperty", "triangles")
            .addParameter("maxDegree", 2)
            .yields();

        assertCypherResult(query, List.of(Map.of(
            "globalTriangleCount", 0L,
            "nodeCount", 5L,
            "createMillis", greaterThan(-1L),
            "computeMillis", greaterThan(-1L),
            "configuration", isA(Map.class),
            "nodePropertiesWritten", 5L,
            "writeMillis", greaterThan(-1L)
        )));

        Map<String, Long> expectedResult = Map.of(
            "a", -1L,
            "b", -1L,
            "c", -1L,
            "d", -1L,
            "e", 0L
        );

        assertWriteResult(expectedResult, "triangles");
    }

    @Override
    public Class<? extends AlgoBaseProc<IntersectingTriangleCount, IntersectingTriangleCount.TriangleCountResult, TriangleCountWriteConfig>> getProcedureClazz() {
        return TriangleCountWriteProc.class;
    }

    @Override
    public TriangleCountWriteConfig createConfig(CypherMapWrapper mapWrapper) {
        return TriangleCountWriteConfig.of(
            getUsername(),
            Optional.empty(),
            Optional.empty(),
            mapWrapper
        );
    }

    @Override
    public CypherMapWrapper createMinimalConfig(CypherMapWrapper mapWrapper) {
        if (!mapWrapper.containsKey("writeProperty")) {
            mapWrapper = mapWrapper.withString("writeProperty", "writeProperty");
        }
        return mapWrapper;
    }

    private void assertWriteResult(
        Map<String, Long> expectedResult,
        String writeProperty
    ) {
        runQueryWithRowConsumer(formatWithLocale(
            "MATCH (n) RETURN n.name AS name, n.%s AS triangles",
            writeProperty
        ), (row) -> {
            long triangles = row.getNumber("triangles").longValue();
            String name = row.getString("name");
            Long expectedTriangles = expectedResult.get(name);
            assertEquals(expectedTriangles, triangles);
        });
    }

}
