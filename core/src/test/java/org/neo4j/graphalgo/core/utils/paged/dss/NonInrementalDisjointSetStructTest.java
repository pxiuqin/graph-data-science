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
package org.neo4j.graphalgo.core.utils.paged.dss;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.core.utils.mem.MemoryRange;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;

import static org.neo4j.graphalgo.core.utils.paged.dss.NonInrementalDisjointSetStruct.memoryEstimation;

class NonInrementalDisjointSetStructTest extends DisjointSetStructTest {

    @Override
    SequentialDisjointSetStruct newSet(final int capacity) {
        AllocationTracker tracker = AllocationTracker.empty();
        return new NonInrementalDisjointSetStruct(
                capacity,
                tracker);
    }

    @Test
    void shouldComputeMemoryEstimation() {
        assertMemoryEstimation(memoryEstimation(), 0, MemoryRange.of(64));
        assertMemoryEstimation(memoryEstimation(), 100, MemoryRange.of(864));
        assertMemoryEstimation(memoryEstimation(), 100_000_000_000L, MemoryRange.of(800_122_070_392L));
    }
}
