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

import org.neo4j.graphalgo.core.utils.mem.MemoryUsage;

public final class HugeArrays {

    static final int PAGE_SHIFT = 14;
    static final int PAGE_SIZE = 1 << PAGE_SHIFT;
    private static final long PAGE_MASK = PAGE_SIZE - 1;

    static int pageIndex(long index) {
        return (int) (index >>> PAGE_SHIFT);
    }

    static int indexInPage(long index) {
        return (int) (index & PAGE_MASK);
    }

    static int exclusiveIndexOfPage(long index) {
        return 1 + (int) ((index - 1L) & PAGE_MASK);
    }

    static long indexFromPageIndexAndIndexInPage(int pageIndex, int indexInPage) {
        return (pageIndex << PAGE_SHIFT) | indexInPage;
    }

    static int numberOfPages(long capacity) {
        final long numPages = (capacity + PAGE_MASK) >>> PAGE_SHIFT;
        assert numPages <= Integer.MAX_VALUE : "pageSize=" + (PAGE_SIZE) + " is too small for capacity: " + capacity;
        return (int) numPages;
    }

    /**
     * Huge version of Lucene oversize for arrays.
     * @see org.apache.lucene.util.ArrayUtil#oversize(int, int)
     */
    public static long oversize(long minTargetSize, int bytesPerElement) {

        if (minTargetSize == 0) {
            // wait until at least one element is requested
            return 0;
        }

        // asymptotic exponential growth by 1/8th, favors
        // spending a bit more CPU to not tie up too much wasted
        // RAM:
        long extra = minTargetSize >> 3;

        if (extra < 3) {
            // for very small arrays, where constant overhead of
            // realloc is presumably relatively high, we grow
            // faster
            extra = 3;
        }

        long newSize = minTargetSize + extra;

        if (MemoryUsage.BYTES_OBJECT_REF == 8) {
            // round up to 8 byte alignment to match JVM pointer size
            switch (bytesPerElement) {
                case 4:
                    // round up to multiple of 2
                    return (newSize + 1) & 0x7FFF_FFFE;
                case 2:
                    // round up to multiple of 4
                    return (newSize + 3) & 0x7FFF_FFFC;
                case 1:
                    // round up to multiple of 8
                    return (newSize + 7) & 0x7FFF_FFF8;
                case 8:
                    // no rounding
                default:
                    return newSize;
            }
        } else {
            // round up to 4 byte alignment to match JVM pointer size
            switch (bytesPerElement) {
                case 2:
                    // round up to multiple of 2
                    return (newSize + 1) & 0x7FFFFFFE;
                case 1:
                    // round up to multiple of 4
                    return (newSize + 3) & 0x7FFFFFFC;
                case 4:
                case 8:
                    // no rounding
                default:
                    return newSize;
            }
        }
    }

    private HugeArrays() {
        throw new UnsupportedOperationException("No instances");
    }
}
