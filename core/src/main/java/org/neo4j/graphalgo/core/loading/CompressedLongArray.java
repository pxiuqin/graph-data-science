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

import org.apache.lucene.util.ArrayUtil;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;

import java.util.Arrays;

import static org.neo4j.graphalgo.core.loading.AdjacencyBuilder.IGNORE_VALUE;
import static org.neo4j.graphalgo.core.loading.VarLongEncoding.encodeVLongs;
import static org.neo4j.graphalgo.core.loading.VarLongEncoding.encodedVLongSize;
import static org.neo4j.graphalgo.core.loading.VarLongEncoding.zigZag;
import static org.neo4j.graphalgo.core.loading.ZigZagLongDecoding.zigZagUncompress;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfByteArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfDoubleArray;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public final class CompressedLongArray {

    private static final byte[] EMPTY_BYTES = new byte[0];

    private final AllocationTracker tracker;
    private byte[] storage;
    private long[][] weights;
    private int pos;
    private long lastValue;
    private int length;

    public CompressedLongArray(AllocationTracker tracker) {
        this(tracker, 0);
    }

    public CompressedLongArray(AllocationTracker tracker, int numberOfProperties) {
        this.tracker = tracker;
        storage = EMPTY_BYTES;
        weights = new long[numberOfProperties][0];
    }

    /**
     * For memory efficiency, we reuse the {@code values}. They cannot be reused after calling this method.
     *
     * @param values values to write
     * @param start  start index in values
     * @param end    end index in values
     */
    public void add(long[] values, int start, int end, int valuesToAdd) {
        // not inlined to avoid field access
        long currentLastValue = this.lastValue;
        long delta;
        long compressedValue;
        int requiredBytes = 0;
        for (int i = start; i < end; i++) {
            if(values[i] == IGNORE_VALUE) {
                continue;
            }

            delta = values[i] - currentLastValue;
            compressedValue = zigZag(delta);
            currentLastValue = values[i];
            values[i] = compressedValue;
            requiredBytes += encodedVLongSize(compressedValue);
        }
        ensureCapacity(this.pos, requiredBytes, this.storage);
        this.pos = encodeVLongs(values, start, end, this.storage, this.pos);

        this.lastValue = currentLastValue;
        this.length += valuesToAdd;
    }

    /**
     * For memory efficiency, we reuse the {@code values}. They cannot be reused after calling this method.
     *
     * @param values        values to write
     * @param allWeights    weights to write
     * @param start         start index in values and weights
     * @param end           end index in values and weights
     * @param valuesToAdd  the actual number of targets to import from this range
     */
    public void add(long[] values, long[][] allWeights, int start, int end, int valuesToAdd) {
        // write weights
        for (int i = 0; i < allWeights.length; i++) {
            long[] weights = allWeights[i];
            addWeights(values, weights, start, end, i, valuesToAdd);
        }

        // write values
        add(values, start, end, valuesToAdd);
    }

    private void addWeights(long[] values, long[] weights, int start, int end, int weightIndex, int weightsToAdd) {
        ensureCapacity(length, weightsToAdd, weightIndex);

        if (weightsToAdd == end - start) {
            System.arraycopy(weights, start, this.weights[weightIndex], this.length, weightsToAdd);
        } else {
            var writePos = length;
            for (int i = 0; i < (end - start); i++) {
                if (values[start + i] != IGNORE_VALUE) {
                    this.weights[weightIndex][writePos++] = weights[start + i];
                }
            }
        }
    }

    void ensureCapacity(int pos, int required, byte[] storage) {
        int targetLength = pos + required;
        if (targetLength < 0) {
            throw new IllegalArgumentException(formatWithLocale(
                "Encountered numeric overflow in internal buffer. Was at position %d and needed to grow by %d.",
                pos,
                required
            ));
        } else if (storage.length <= targetLength) {
            int newLength = ArrayUtil.oversize(targetLength, Byte.BYTES);
            tracker.remove(sizeOfByteArray(storage.length));
            tracker.add(sizeOfByteArray(newLength));
            this.storage = Arrays.copyOf(storage, newLength);
        }
    }

    private void ensureCapacity(int pos, int required, int weightIndex) {
        int targetLength = pos + required;
        if (targetLength < 0) {
            throw new IllegalArgumentException(formatWithLocale(
                "Encountered numeric overflow in internal buffer. Was at position %d and needed to grow by %d.",
                pos,
                required
            ));
        } else if (weights[weightIndex].length <= pos + required) {
            int newLength = ArrayUtil.oversize(pos + required, Long.BYTES);
            tracker.remove(sizeOfDoubleArray(weights[weightIndex].length));
            tracker.add(sizeOfDoubleArray(newLength));
            weights[weightIndex] = Arrays.copyOf(weights[weightIndex], newLength);
        }
    }

    public int length() {
        return length;
    }

    public int uncompress(long[] into) {
        assert into.length >= length;
        return zigZagUncompress(storage, pos, into);
    }

    public byte[] storage() {
        return storage;
    }

    long[][] weights() {
        return weights;
    }

    public boolean hasWeights() {
        return weights != null && !(weights.length == 0);
    }

    public void release() {
        if (storage.length > 0) {
            tracker.remove(sizeOfByteArray(storage.length));
            tracker.remove(sizeOfDoubleArray(weights.length));
        }
        storage = null;
        weights = null;
        pos = 0;
        length = 0;
    }
}
