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
package org.neo4j.graphalgo.core.loading;

import com.carrotsearch.hppc.IntObjectHashMap;
import com.carrotsearch.hppc.IntObjectMap;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.utils.mem.MemoryRange;
import org.neo4j.graphalgo.core.utils.mem.MemoryTree;

import java.util.Arrays;
import java.util.List;

import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.assertEquals;

class IdMapTest {

    @Test
    void shouldComputeMemoryEstimation() {
        GraphDimensions dimensions = ImmutableGraphDimensions.builder().nodeCount(0).highestNeoId(0).build();
        MemoryTree memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 40L + 40L), memRec.memoryUsage());

        dimensions = ImmutableGraphDimensions.builder().nodeCount(100L).highestNeoId(100L).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 840L + 32832L), memRec.memoryUsage());

        dimensions = ImmutableGraphDimensions.builder().nodeCount(1L).highestNeoId(100_000_000_000L).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 48L + 97_689_080L), memRec.memoryUsage());

        dimensions = ImmutableGraphDimensions.builder().nodeCount(10_000_000L).highestNeoId(100_000_000_000L).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 80_000_040L + 177_714_824L, 40L + 80_000_040L + 327_937_656_296L), memRec.memoryUsage());

        dimensions = ImmutableGraphDimensions.builder().nodeCount(100_000_000L).highestNeoId(100_000_000_000L).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 800_000_040L + 898_077_656L, 40L + 800_000_040L + 800_488_297_688L), memRec.memoryUsage());


        IntObjectMap<List<NodeLabel>> labelTokenNodeLabelMappings = new IntObjectHashMap<List<NodeLabel>>();
        labelTokenNodeLabelMappings.put(1, singletonList(NodeLabel.of("A")));

        dimensions = ImmutableGraphDimensions.builder().nodeCount(100L).highestNeoId(100L)
            .tokenNodeLabelMapping(labelTokenNodeLabelMappings).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 840L + 32832L + 56L), memRec.memoryUsage());

        labelTokenNodeLabelMappings.put(2, Arrays.asList(NodeLabel.of("A"), NodeLabel.of("B")));
        dimensions = ImmutableGraphDimensions.builder().nodeCount(100L).highestNeoId(100L)
            .tokenNodeLabelMapping(labelTokenNodeLabelMappings).build();
        memRec = IdMap.memoryEstimation().estimate(dimensions, 1);
        assertEquals(MemoryRange.of(40L + 840L + 32832L + 112L), memRec.memoryUsage());
    }
}
