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
package org.neo4j.graphalgo.core.utils.paged;

public final class HugeLongArrayStack {

    private final HugeLongArray array;
    private final long capacity;
    private long size;

    public static HugeLongArrayStack newStack(long capacity, AllocationTracker tracker) {
        return new HugeLongArrayStack(HugeLongArray.newArray(capacity, tracker));
    }

    private HugeLongArrayStack(HugeLongArray array) {
        this.capacity = array.size();
        this.array = array;
    }

    public void push(long v) {
        if (size == capacity) {
            throw new IndexOutOfBoundsException("Stack is full.");
        }
        array.set(size++, v);
    }

    public long pop() {
        if (isEmpty()) {
            throw new IndexOutOfBoundsException("Stack is empty.");
        }
        return array.get(--size);
    }

    public long size() {
        return size;
    }

    public boolean isEmpty() {
        return size() == 0;
    }
}
