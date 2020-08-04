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

import org.neo4j.graphalgo.api.AdjacencyCursor;

import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import static org.neo4j.graphalgo.core.huge.TransientAdjacencyList.DecompressingCursor.NOT_FOUND;

public class CompositeAdjacencyCursor implements AdjacencyCursor {

    private final PriorityQueue<AdjacencyCursor> cursorQueue;
    private final List<AdjacencyCursor> cursors;


    CompositeAdjacencyCursor(List<AdjacencyCursor> cursors) {
        this.cursors = cursors;
        this.cursorQueue = new PriorityQueue<>(cursors.size(), Comparator.comparingLong(AdjacencyCursor::peekVLong));

        initializeQueue();
    }

    private void initializeQueue() {
        cursors.forEach(cursor -> {
            if (cursor != null && cursor.hasNextVLong()) {
                cursorQueue.add(cursor);
            }
        });
    }

    public List<AdjacencyCursor> cursors() {
        return cursors;
    }

    void copyFrom(CompositeAdjacencyCursor other) {
        List<AdjacencyCursor> otherCursors = other.cursors();
        for (int i = 0; i < cursors.size(); i++) {
            var cursor = (TransientAdjacencyList.DecompressingCursor) cursors.get(i);
            var otherCursor = (TransientAdjacencyList.DecompressingCursor) otherCursors.get(i);
            cursor.copyFrom(otherCursor);
        }
    }

    @Override
    public int size() {
        return cursors.stream().mapToInt(AdjacencyCursor::size).sum();
    }

    @Override
    public boolean hasNextVLong() {
        return !cursorQueue.isEmpty();
    }

    @Override
    public long nextVLong() {
        AdjacencyCursor current = cursorQueue.poll();
        long targetNodeId = current.nextVLong();
        if (current.hasNextVLong()) {
            cursorQueue.add(current);
        }
        return targetNodeId;
    }

    @Override
    public long peekVLong() {
        return cursorQueue.peek().peekVLong();
    }

    @Override
    public int remaining() {
        return cursors.stream().mapToInt(AdjacencyCursor::remaining).sum();
    }

    @Override
    public void close() {
        cursors.forEach(AdjacencyCursor::close);
    }

    long skipUntil(long target) {
        for (var cursor : cursors) {
            cursorQueue.remove(cursor);
            // an implementation aware cursor would probably be much faster and could skip whole blocks
            // see AdjacencyDecompressingReader#skipUntil
            while (cursor.hasNextVLong() && cursor.peekVLong() <= target) {
                cursor.nextVLong();
            }
            if (cursor.hasNextVLong()) {
                cursorQueue.add(cursor);
            }
        }

        return cursorQueue.isEmpty() ? NOT_FOUND : nextVLong();
    }

    long advance(long target) {
        for (var cursor : cursors) {
            cursorQueue.remove(cursor);
            // an implementation aware cursor would probably be much faster and could skip whole blocks
            // see AdjacencyDecompressingReader#advance
            while (cursor.hasNextVLong() && cursor.peekVLong() < target) {
                cursor.nextVLong();
            }
            if (cursor.hasNextVLong()) {
                cursorQueue.add(cursor);
            }
        }

        return cursorQueue.isEmpty() ? NOT_FOUND : nextVLong();
    }
}
