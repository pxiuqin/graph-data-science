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
package org.neo4j.graphalgo.wcc;

import com.carrotsearch.hppc.BitSet;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.neo4j.graphalgo.AlgoTestBase;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.StoreLoaderBuilder;
import org.neo4j.graphalgo.TestLog;
import org.neo4j.graphalgo.TestProgressLogger;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.mem.MemoryRange;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.dss.DisjointSetStruct;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Transaction;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.compat.GraphDatabaseApiProxy.runInTransaction;

/**
 * 弱连通分量测试
 */
class WccTest extends AlgoTestBase {

    static final RelationshipType RELATIONSHIP_TYPE = RelationshipType.withName("TYPE");
    private static final int SETS_COUNT = 16;
    private static final int SET_SIZE = 10;

    /**
     * Compute number of sets present.
     */
    static long getSetCount(DisjointSetStruct struct) {
        long capacity = struct.size();
        BitSet sets = new BitSet(capacity);
        for (long i = 0L; i < capacity; i++) {
            long setId = struct.setIdOf(i);
            sets.set(setId);
        }
        return sets.cardinality();
    }

    @BeforeEach
    void setupGraphDb() {
        int[] setSizes = new int[SETS_COUNT];
        Arrays.fill(setSizes, SET_SIZE);
        createTestGraph(setSizes);
    }

    int communitySize() {
        return SET_SIZE;
    }

    @ParameterizedTest(name = "orientation = {1}")
    @EnumSource(Orientation.class)
    void shouldComputeComponents(Orientation orientation) {
        Graph graph = new StoreLoaderBuilder()
            .api(db)
            .addRelationshipType(RELATIONSHIP_TYPE.name())
            .globalOrientation(orientation)
            .build()
            .graph();

        DisjointSetStruct result = run(graph);

        assertEquals(SETS_COUNT, getSetCount(result));
        long[] setRegions = new long[SETS_COUNT];
        Arrays.fill(setRegions, -1);

        graph.forEachNode((nodeId) -> {
            long expectedSetRegion = nodeId / SET_SIZE;
            final long setId = result.setIdOf(nodeId);
            int setRegion = (int) (setId / SET_SIZE);
            assertEquals(
                expectedSetRegion,
                setRegion,
                "Node " + nodeId + " in unexpected set: " + setId
            );

            long regionSetId = setRegions[setRegion];
            if (regionSetId == -1) {
                setRegions[setRegion] = setId;
            } else {
                assertEquals(
                    regionSetId,
                    setId,
                    "Inconsistent set for node " + nodeId + ", is " + setId + " but should be " + regionSetId
                );
            }
            return true;
        });
    }

    @Test
    void shouldLogProgress() {
        var graph = new StoreLoaderBuilder()
            .api(db)
            .addRelationshipType(RELATIONSHIP_TYPE.name())
            .globalOrientation(Orientation.NATURAL)
            .build()
            .graph();

        var testLogger = new TestProgressLogger(graph.relationshipCount(), "Wcc", 2);

        new Wcc(
            graph,
            Pools.DEFAULT,
            communitySize() / 4,
            ImmutableWccStreamConfig.builder().concurrency(2).build(),
            testLogger,
            AllocationTracker.EMPTY
        ).compute();

        List<AtomicLong> progresses = testLogger.getProgresses();
        assertEquals(1, progresses.size());
        assertEquals(graph.relationshipCount(), progresses.get(0).get());

        assertTrue(testLogger.containsMessage(TestLog.INFO, ":: Start"));
        assertTrue(testLogger.containsMessage(TestLog.INFO, ":: Finished"));
    }

    @Test
    void memRecParallel() {
        GraphDimensions dimensions0 = ImmutableGraphDimensions.builder().nodeCount(0).build();

        assertEquals(
            MemoryRange.of(128),
            Wcc.memoryEstimation(false).estimate(dimensions0, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(168),
            Wcc.memoryEstimation(true).estimate(dimensions0, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(128),
            Wcc.memoryEstimation(false).estimate(dimensions0, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(168),
            Wcc.memoryEstimation(true).estimate(dimensions0, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(128),
            Wcc.memoryEstimation(false).estimate(dimensions0, 64).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(168),
            Wcc.memoryEstimation(true).estimate(dimensions0, 64).memoryUsage()
        );

        GraphDimensions dimensions100 = ImmutableGraphDimensions.builder().nodeCount(100).build();
        assertEquals(
            MemoryRange.of(928),
            Wcc.memoryEstimation(false).estimate(dimensions100, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1768),
            Wcc.memoryEstimation(true).estimate(dimensions100, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(928),
            Wcc.memoryEstimation(false).estimate(dimensions100, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1768),
            Wcc.memoryEstimation(true).estimate(dimensions100, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(928),
            Wcc.memoryEstimation(false).estimate(dimensions100, 64).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1768),
            Wcc.memoryEstimation(true).estimate(dimensions100, 64).memoryUsage()
        );

        GraphDimensions dimensions100B = ImmutableGraphDimensions.builder().nodeCount(100_000_000_000L).build();
        assertEquals(
            MemoryRange.of(800_122_070_456L),
            Wcc.memoryEstimation(false).estimate(dimensions100B, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1_600_244_140_824L),
            Wcc.memoryEstimation(true).estimate(dimensions100B, 1).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(800_122_070_456L),
            Wcc.memoryEstimation(false).estimate(dimensions100B, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1_600_244_140_824L),
            Wcc.memoryEstimation(true).estimate(dimensions100B, 8).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(800_122_070_456L),
            Wcc.memoryEstimation(false).estimate(dimensions100B, 64).memoryUsage()
        );
        assertEquals(
            MemoryRange.of(1_600_244_140_824L),
            Wcc.memoryEstimation(true).estimate(dimensions100B, 64).memoryUsage()
        );
    }

    private void createTestGraph(int... setSizes) {
        runInTransaction(db, tx -> {
            for (int setSize : setSizes) {
                createLine(db, tx, setSize);
            }
        });
    }

    private static void createLine(GraphDatabaseService db, Transaction tx, int setSize) {
        Node temp = tx.createNode();
        for (int i = 1; i < setSize; i++) {
            Node t = tx.createNode();
            temp.createRelationshipTo(t, RELATIONSHIP_TYPE);
            temp = t;
        }
    }

    DisjointSetStruct run(Graph graph) {
        return run(graph, ImmutableWccStreamConfig.builder().build());
    }

    DisjointSetStruct run(Graph graph, WccBaseConfig config) {
        return run(graph, config, communitySize() / config.concurrency());
    }

    DisjointSetStruct run(Graph graph, WccBaseConfig config, int concurrency) {
        return new Wcc(
            graph,
            Pools.DEFAULT,
            communitySize() / concurrency,
            config,
            progressLogger,
            AllocationTracker.EMPTY
        ).compute();
    }
}
