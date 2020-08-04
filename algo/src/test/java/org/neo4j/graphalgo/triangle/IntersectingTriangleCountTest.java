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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.triangle.IntersectingTriangleCount.TriangleCountResult;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.Orientation.UNDIRECTED;
import static org.neo4j.graphalgo.TestSupport.fromGdl;
import static org.neo4j.graphalgo.triangle.IntersectingTriangleCount.EXCLUDED_NODE_TRIANGLE_COUNT;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

class IntersectingTriangleCountTest {

    private static Stream<Arguments> noTriangleQueries() {
        return Stream.of(
            Arguments.of(fromGdl("CREATE ()-[:T]->()-[:T]->()", UNDIRECTED), "line"),
            Arguments.of(fromGdl("CREATE (), (), ()", UNDIRECTED), "no rels"),
            Arguments.of(fromGdl("CREATE ()-[:T]->(), ()", UNDIRECTED), "one rel"),
            Arguments.of(fromGdl("CREATE (a1)-[:T]->()-[:T]->(a1), ()", UNDIRECTED), "back and forth")
        );
    }

    @MethodSource("noTriangleQueries")
    @ParameterizedTest(name = "{1}")
    void noTriangles(Graph graph, String ignoredName) {
        TriangleCountResult result = compute(graph);

        assertEquals(0L, result.globalTriangles());
        assertEquals(3, result.localTriangles().size());
        assertEquals(0, result.localTriangles().get(0));
        assertEquals(0, result.localTriangles().get(1));
        assertEquals(0, result.localTriangles().get(2));
    }

    @ValueSource(ints = {1, 2, 4, 8, 100})
    @ParameterizedTest
    void independentTriangles(int nbrOfTriangles) {
        StringBuilder gdl = new StringBuilder("CREATE ");
        for (int i = 0; i < nbrOfTriangles; ++i) {
            gdl.append(formatWithLocale("(a%d)-[:T]->()-[:T]->()-[:T]->(a%d) ", i, i));
        }

        TriangleCountResult result = compute(fromGdl(gdl.toString(), UNDIRECTED));

        assertEquals(nbrOfTriangles, result.globalTriangles());
        assertEquals(3 * nbrOfTriangles, result.localTriangles().size());
        for (int i = 0; i < result.localTriangles().size(); ++i) {
            assertEquals(1, result.localTriangles().get(i));
        }
    }

    @Test
    void clique5() {
        var graph = fromGdl(
            "CREATE " +
            " (a1)-[:T]->(a2), " +
            " (a1)-[:T]->(a3), " +
            " (a1)-[:T]->(a4), " +
            " (a1)-[:T]->(a5), " +
            " (a2)-[:T]->(a3), " +
            " (a2)-[:T]->(a4), " +
            " (a2)-[:T]->(a5), " +
            " (a3)-[:T]->(a4), " +
            " (a3)-[:T]->(a5), " +
            " (a4)-[:T]->(a5)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(10, result.globalTriangles());
        assertEquals(5, result.localTriangles().size());
        for (int i = 0; i < result.localTriangles().size(); ++i) {
            assertEquals(6, result.localTriangles().get(i));
        }
    }

    @Test
    void clique5UnionGraph() {
        var graph = fromGdl(
            "CREATE " +
            " (a1)-[:T1]->(a2), " +
            " (a1)-[:T1]->(a3), " +
            " (a1)-[:T2]->(a4), " +
            " (a1)-[:T3]->(a5), " +
            " (a2)-[:T4]->(a3), " +
            " (a2)-[:T2]->(a4), " +
            " (a2)-[:T2]->(a5), " +
            " (a3)-[:T3]->(a4), " +
            " (a3)-[:T1]->(a5), " +
            " (a4)-[:T4]->(a5)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(10, result.globalTriangles());
        assertEquals(5, result.localTriangles().size());
        for (int i = 0; i < result.localTriangles().size(); ++i) {
            assertEquals(6, result.localTriangles().get(i));
        }
    }

    @Test
    void twoAdjacentTriangles() {
        var graph = fromGdl(
            "CREATE " +
            "  (a)-[:T]->()-[:T]->()-[:T]->(a) " +
            ", (a)-[:T]->()-[:T]->()-[:T]->(a)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(2, result.globalTriangles());
        assertEquals(5, result.localTriangles().size());
    }

    @Test
    void twoTrianglesWithLine() {
        var graph = fromGdl(
            "CREATE " +
            "  (a)-[:T]->(b)-[:T]->(c)-[:T]->(a) " +
            ", (q)-[:T]->(r)-[:T]->(t)-[:T]->(q) " +
            ", (a)-[:T]->(q)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(2, result.globalTriangles());
        assertEquals(6, result.localTriangles().size());

        for (int i = 0; i < result.localTriangles().size(); ++i) {
            assertEquals(1, result.localTriangles().get(i));
        }
    }

    @Test
    void selfLoop() {
        var graph = fromGdl("CREATE (a)-[:T]->(a)-[:T]->(a)-[:T]->(a)", UNDIRECTED);

        TriangleCountResult result = compute(graph);

        assertEquals(0, result.globalTriangles());
        assertEquals(1, result.localTriangles().size());
        assertEquals(0, result.localTriangles().get(0));
    }

    @Test
    void selfLoop2() {
        var graph = fromGdl("CREATE (a)-[:T]->(b)-[:T]->(c)-[:T]->(a)-[:T]->(a)", UNDIRECTED);

        TriangleCountResult result = compute(graph);

        assertEquals(1, result.globalTriangles());
        assertEquals(3, result.localTriangles().size());
        assertEquals(1, result.localTriangles().get(0));
        assertEquals(1, result.localTriangles().get(1));
        assertEquals(1, result.localTriangles().get(2));
    }

    @Test
    void parallelRelationships() {
        var graph = fromGdl(
            "CREATE" +
            " (a)-[:T]->(b)-[:T]->(c)-[:T]->(a)" +
            ", (a)-[:T]->(b)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(1, result.globalTriangles());
        assertEquals(3, result.localTriangles().size());
        assertEquals(1, result.localTriangles().get(0));
        assertEquals(1, result.localTriangles().get(1));
        assertEquals(1, result.localTriangles().get(2));
    }

    @Test
    void parallelTriangles() {
        var graph = fromGdl(
            "CREATE" +
            " (a)-[:T]->(b)-[:T]->(c)-[:T]->(a)" +
            ",(a)-[:T]->(b)-[:T]->(c)-[:T]->(a)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(1, result.globalTriangles());
        assertEquals(3, result.localTriangles().size());
        assertEquals(1, result.localTriangles().get(0));
        assertEquals(1, result.localTriangles().get(1));
        assertEquals(1, result.localTriangles().get(2));
    }

    @Test
    void manyTrianglesAndOtherThings() {
        var graph = fromGdl(
            "CREATE" +
            " (a)-[:T]->(b)-[:T]->(b)-[:T]->(c)-[:T]->(a)" +
            ", (c)-[:T]->(d)-[:T]->(e)-[:T]->(f)-[:T]->(d)" +
            ", (f)-[:T]->(g)-[:T]->(h)-[:T]->(f)" +
            ", (h)-[:T]->(i)-[:T]->(j)-[:T]->(k)-[:T]->(e)" +
            ", (k)-[:T]->(l)" +
            ", (k)-[:T]->(m)-[:T]->(n)-[:T]->(j)" +
            ", (o)",
            UNDIRECTED
        );

        TriangleCountResult result = compute(graph);

        assertEquals(3, result.globalTriangles());
        assertEquals(15, result.localTriangles().size());
        assertEquals(1, result.localTriangles().get(0)); // a
        assertEquals(1, result.localTriangles().get(1)); // b
        assertEquals(1, result.localTriangles().get(2)); // c
        assertEquals(1, result.localTriangles().get(3)); // d
        assertEquals(1, result.localTriangles().get(4)); // e
        assertEquals(2, result.localTriangles().get(5)); // f
        assertEquals(1, result.localTriangles().get(6)); // g
        assertEquals(1, result.localTriangles().get(7)); // h
        assertEquals(0, result.localTriangles().get(8)); // i
        assertEquals(0, result.localTriangles().get(9)); // j
        assertEquals(0, result.localTriangles().get(10)); // k
        assertEquals(0, result.localTriangles().get(11)); // l
        assertEquals(0, result.localTriangles().get(12)); // m
        assertEquals(0, result.localTriangles().get(13)); // n
        assertEquals(0, result.localTriangles().get(14)); // o
    }

    @Test
    void testTriangleCountingWithMaxDegree() {
        var graph = fromGdl(
            "CREATE" +
            "  (a)-[:T]->(b)"+
            " ,(a)-[:T]->(c)"+
            " ,(a)-[:T]->(d)"+
            " ,(b)-[:T]->(c)"+
            " ,(b)-[:T]->(d)"+

            " ,(e)-[:T]->(f)"+
            " ,(f)-[:T]->(g)"+
            " ,(g)-[:T]->(e)",
            UNDIRECTED
        );

        TriangleCountBaseConfig config = ImmutableTriangleCountBaseConfig
            .builder()
            .maxDegree(2)
            .build();

        TriangleCountResult result = compute(graph, config);

        assertEquals(EXCLUDED_NODE_TRIANGLE_COUNT, result.localTriangles().get(0)); // a (deg = 3)
        assertEquals(EXCLUDED_NODE_TRIANGLE_COUNT, result.localTriangles().get(1)); // b (deg = 3)
        assertEquals(0, result.localTriangles().get(2));  // c (deg = 2)
        assertEquals(0, result.localTriangles().get(3));  // d (deg = 2)

        assertEquals(1, result.localTriangles().get(4)); // e (deg = 2)
        assertEquals(1, result.localTriangles().get(5)); // f (deg = 2)
        assertEquals(1, result.localTriangles().get(6)); // g (deg = 2)
        assertEquals(1, result.globalTriangles());
    }

    @Test
    void testTriangleCountingWithMaxDegreeOnUnionGraph() {
        var graph = fromGdl(
            "CREATE" +
            "  (a)-[:T1]->(b)"+
            " ,(a)-[:T2]->(c)"+
            " ,(a)-[:T2]->(d)"+
            " ,(b)-[:T1]->(c)"+
            " ,(b)-[:T2]->(d)"+

            " ,(e)-[:T1]->(f)"+
            " ,(f)-[:T1]->(g)"+
            " ,(g)-[:T1]->(e)",
            UNDIRECTED
        );

        TriangleCountBaseConfig config = ImmutableTriangleCountBaseConfig
            .builder()
            .maxDegree(2)
            .build();

        TriangleCountResult result = compute(graph, config);

        assertEquals(EXCLUDED_NODE_TRIANGLE_COUNT, result.localTriangles().get(0)); // a (deg = 3)
        assertEquals(EXCLUDED_NODE_TRIANGLE_COUNT, result.localTriangles().get(1)); // b (deg = 3)
        assertEquals(0, result.localTriangles().get(2));  // c (deg = 2)
        assertEquals(0, result.localTriangles().get(3));  // d (deg = 2)

        assertEquals(1, result.localTriangles().get(4)); // e (deg = 2)
        assertEquals(1, result.localTriangles().get(5)); // f (deg = 2)
        assertEquals(1, result.localTriangles().get(6)); // g (deg = 2)
        assertEquals(1, result.globalTriangles());
    }

    private TriangleCountResult compute(Graph graph) {
        TriangleCountStatsConfig config = ImmutableTriangleCountStatsConfig.builder().build();
        return compute(graph, config);
    }

    private TriangleCountResult compute(Graph graph, TriangleCountBaseConfig config) {
        return new IntersectingTriangleCount(
            graph,
            config,
            Pools.DEFAULT,
            AllocationTracker.EMPTY
        ).compute();
    }
}
