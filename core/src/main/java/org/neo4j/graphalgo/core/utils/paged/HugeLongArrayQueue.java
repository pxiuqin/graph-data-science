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

import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;

public final class HugeLongArrayQueue {

    private final HugeLongArray array;
    private final long capacity;
    private long head;
    private long tail;

    public static HugeLongArrayQueue newQueue(long capacity, AllocationTracker tracker) {
        return new HugeLongArrayQueue(HugeLongArray.newArray(capacity + 1, tracker));
    }

    private HugeLongArrayQueue(HugeLongArray array) {
        this.head = 0;
        this.tail = 0;
        this.capacity = array.size();
        this.array = array;
    }

    public void add(long v) {
        long newTail = (tail + 1) % capacity;
        if (newTail == head) {
            throw new IndexOutOfBoundsException("Queue is full.");
        }
        array.set(tail, v);
        tail = newTail;
    }

    public long remove() {
        if (isEmpty()) {
            throw new IndexOutOfBoundsException("Queue is empty.");
        }
        long removed = array.get(head);
        head = (head + 1) % capacity;
        return removed;
    }

    public long size() {
        long diff = tail - head;
        if (diff < 0) {
            diff += capacity;
        }
        return diff;
    }

    public boolean isEmpty() {
        return head == tail;
    }
}
