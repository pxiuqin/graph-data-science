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

import org.neo4j.graphalgo.core.loading.StoreScanner.RecordConsumer;

public abstract class RecordsBatchBuffer<Reference> implements RecordConsumer<Reference> {

    static final int DEFAULT_BUFFER_SIZE = 100_000;

    final long[] buffer;
    int length;

    RecordsBatchBuffer(int capacity) {
        this.buffer = new long[capacity];
    }

    public boolean scan(StoreScanner.ScanCursor<Reference> cursor) {
        reset();
        return cursor.bulkNext(this);
    }

    public int length() {
        return length;
    }

    int capacity() {
        return buffer.length;
    }

    public boolean isFull() {
        return length >= buffer.length;
    }

    public void reset() {
        this.length = 0;
    }

    public long[] batch() {
        return buffer;
    }

}
