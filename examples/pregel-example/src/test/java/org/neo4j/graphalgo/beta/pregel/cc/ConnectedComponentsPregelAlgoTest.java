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
package org.neo4j.graphalgo.beta.pregel.cc;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.TestSupport;
import org.neo4j.graphalgo.beta.pregel.Pregel;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.Inject;
import org.neo4j.graphalgo.extension.TestGraph;

import java.util.HashMap;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.beta.pregel.cc.ConnectedComponentsPregel.COMPONENT;
import static org.neo4j.graphalgo.core.ExceptionMessageMatcher.containsMessage;

@GdlExtension
class ConnectedComponentsPregelAlgoTest {

    @GdlGraph(graphNamePrefix = "directed")
    private static final String TEST_GRAPH =
            "CREATE" +
            "  (a:Node)" +
            ", (b:Node)" +
            ", (c:Node)" +
            ", (d:Node)" +
            ", (e:Node)" +
            ", (f:Node)" +
            ", (g:Node)" +
            ", (h:Node)" +
            ", (i:Node)" +
            // {J}
            ", (j:Node)" +
            // {A, B, C, D}
            ", (a)-[:TYPE]->(b)" +
            ", (b)-[:TYPE]->(c)" +
            ", (c)-[:TYPE]->(d)" +
            ", (d)-[:TYPE]->(a)" +
            // {E, F, G}
            ", (e)-[:TYPE]->(f)" +
            ", (f)-[:TYPE]->(g)" +
            ", (g)-[:TYPE]->(e)" +
            // {H, I}
            ", (i)-[:TYPE]->(h)" +
            ", (h)-[:TYPE]->(i)";

    @GdlGraph(graphNamePrefix = "undirected", orientation = Orientation.UNDIRECTED)
    private static final String TEST_GRAPH_UNDIRECTED = TEST_GRAPH;

    @Inject
    private TestGraph directedGraph;

    @Inject
    private TestGraph undirectedGraph;

    @Test
    void directedSCC() {
        int maxIterations = 10;

        var config = ImmutableConnectedComponentsConfig.builder()
            .maxIterations(maxIterations)
            .build();

        var pregelJob = Pregel.create(
            directedGraph,
            config,
            new ConnectedComponentsPregel(),
            Pools.DEFAULT,
            AllocationTracker.empty()
        );

        var result = pregelJob.run();

        assertTrue(result.didConverge(), "Algorithm did not converge.");
        assertEquals(3, result.ranIterations());

        var expected = new HashMap<String, Long>();
        expected.put("a", 0L);
        expected.put("b", 0L);
        expected.put("c", 0L);
        expected.put("d", 0L);
        expected.put("e", 4L);
        expected.put("f", 4L);
        expected.put("g", 4L);
        expected.put("h", 7L);
        expected.put("i", 7L);
        expected.put("j", 9L);

        TestSupport.assertLongValues(directedGraph, (nodeId) -> result.nodeValues().longValue(COMPONENT, nodeId), expected);
    }

    @Test
    void undirectedWCC() {
        int maxIterations = 10;

        var config = ImmutableConnectedComponentsConfig.builder()
            .concurrency(2)
            .maxIterations(maxIterations)
            .build();

        var pregelJob = Pregel.create(
            undirectedGraph,
            config,
            new ConnectedComponentsPregel(),
            Pools.DEFAULT,
            AllocationTracker.empty()
        );

        var result = pregelJob.run();

        assertTrue(result.didConverge(), "Algorithm did not converge.");
        assertEquals(3, result.ranIterations());

        var expected = new HashMap<String, Long>();
        expected.put("a", 0L);
        expected.put("b", 0L);
        expected.put("c", 0L);
        expected.put("d", 0L);
        expected.put("e", 4L);
        expected.put("f", 4L);
        expected.put("g", 4L);
        expected.put("h", 7L);
        expected.put("i", 7L);
        expected.put("j", 9L);

        TestSupport.assertLongValues(undirectedGraph, (nodeId) -> result.nodeValues().longValue(COMPONENT, nodeId), expected);
    }

    @Test
    void shouldFailWithConcurrency10() {
        int maxIterations = 10;

        IllegalArgumentException illegalArgumentException = assertThrows(IllegalArgumentException.class, () -> {
            ImmutableConnectedComponentsConfig.builder()
                .concurrency(10)
                .maxIterations(maxIterations)
                .build();
        });

        assertThat(illegalArgumentException, containsMessage("The configured `writeConcurrency` value is too high"));
    }
}
