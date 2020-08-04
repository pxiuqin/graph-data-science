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

import org.neo4j.graphalgo.core.loading.MutableIntValue;

import java.util.Arrays;

import static org.neo4j.graphalgo.core.huge.VarLongDecoding.decodeDeltaVLongs;

final class AdjacencyDecompressingReader {

    static final int CHUNK_SIZE = 64;

    private final long[] block;
    private int pos;
    private byte[] array;
    private int offset;

    private boolean blockAlreadyDecoded;

    AdjacencyDecompressingReader() {
        this.block = new long[CHUNK_SIZE];
        this.blockAlreadyDecoded = false;
    }

    //@formatter:off
    static int readInt(byte[] array, int offset) {
        return   array[    offset] & 255        |
                (array[1 + offset] & 255) <<  8 |
                (array[2 + offset] & 255) << 16 |
                (array[3 + offset] & 255) << 24;
    }
    //@formatter:on

    //@formatter:off
    static long readLong(byte[] array, int offset) {
        return   array[    offset] & 255L        |
                (array[1 + offset] & 255L) <<  8 |
                (array[2 + offset] & 255L) << 16 |
                (array[3 + offset] & 255L) << 24 |
                (array[4 + offset] & 255L) << 32 |
                (array[5 + offset] & 255L) << 40 |
                (array[6 + offset] & 255L) << 48 |
                (array[7 + offset] & 255L) << 56;
    }
    //@formatter:on

    void copyFrom(AdjacencyDecompressingReader other) {
        System.arraycopy(other.block, 0, block, 0, CHUNK_SIZE);
        pos = other.pos;
        array = other.array;
        offset = other.offset;
    }

    int reset(byte[] adjacencyPage, int offset) {
        this.array = adjacencyPage;
        int numAdjacencies = readInt(adjacencyPage, offset); // offset should not be 0
        this.offset = decodeDeltaVLongs(0L, adjacencyPage, Integer.BYTES + offset, Math.min(numAdjacencies, CHUNK_SIZE), block);
        pos = 0;
        return numAdjacencies;
    }

    long next(int remaining) {
        int pos = this.pos++;
        if (pos < CHUNK_SIZE) {
            return block[pos];
        }
        long targetNode = readNextBlock(remaining);
        this.pos = 1;
        return targetNode;
    }

    long peek(int remaining) {
        int pos = this.pos;
        if (pos < CHUNK_SIZE) {
            return block[pos];
        }
        long targetNode = readNextBlock(remaining);
        blockAlreadyDecoded = true;
        this.pos = 0;
        return targetNode;
    }

    private long readNextBlock(int remaining) {
        if (!blockAlreadyDecoded) {
            offset = decodeDeltaVLongs(block[CHUNK_SIZE - 1], array, offset, Math.min(remaining, CHUNK_SIZE), block);
            return block[0];
        }
        blockAlreadyDecoded = false;
        return block[pos];
    }

    long skipUntil(long target, int remaining, MutableIntValue consumed) {
        int pos = this.pos;
        long[] block = this.block;
        int available = remaining;

        // skip blocks until we have either not enough available to decode or have advanced far enough
        while (available > CHUNK_SIZE - pos && block[CHUNK_SIZE - 1] <= target) {
            int skippedInThisBlock = CHUNK_SIZE - pos;
            int needToDecode = Math.min(CHUNK_SIZE, available - skippedInThisBlock);
            offset = decodeDeltaVLongs(block[CHUNK_SIZE - 1], array, offset, needToDecode, block);
            available -= skippedInThisBlock;
            pos = 0;
        }

        // last block
        if(available <= 0) {
            return TransientAdjacencyList.DecompressingCursor.NOT_FOUND;
        }

        int targetPos = findPosStrictlyGreaterInBlock(target, pos, Math.min(pos + available, CHUNK_SIZE), block);
        // we need to consume including targetPos, not to it, therefore +1
        available -= (1 + targetPos - pos);
        consumed.value = remaining - available;
        this.pos = 1 + targetPos;
        return block[targetPos];
    }

    long advance(long target, int remaining, MutableIntValue consumed) {
        int pos = this.pos;
        long[] block = this.block;
        int available = remaining;

        // skip blocks until we have either not enough available to decode or have advanced far enough
        while (available > CHUNK_SIZE - pos && block[CHUNK_SIZE - 1] < target) {
            int skippedInThisBlock = CHUNK_SIZE - pos;
            int needToDecode = Math.min(CHUNK_SIZE, available - skippedInThisBlock);
            offset = decodeDeltaVLongs(block[CHUNK_SIZE - 1], array, offset, needToDecode, block);
            available -= skippedInThisBlock;
            pos = 0;
        }

        // last block
        int targetPos = findPosInBlock(target, pos, Math.min(pos + available, CHUNK_SIZE), block);
        // we need to consume including targetPos, not to it, therefore +1
        available -= (1 + targetPos - pos);
        consumed.value = remaining - available;
        this.pos = 1 + targetPos;
        return block[targetPos];
    }

    private int findPosStrictlyGreaterInBlock(long target, int pos, int limit, long[] block) {
        return findPosInBlock(1L + target, pos, limit, block);
    }

    private int findPosInBlock(long target, int pos, int limit, long[] block) {
        int targetPos = Arrays.binarySearch(block, pos, limit, target);
        if (targetPos < 0) {
            targetPos = Math.min(-1 - targetPos, -1 + limit);
        }
        return targetPos;
    }
}
