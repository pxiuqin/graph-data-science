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

import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.AdjacencyCursor;
import org.neo4j.graphalgo.api.AdjacencyList;
import org.neo4j.graphalgo.api.PropertyCursor;
import org.neo4j.graphalgo.core.loading.MutableIntValue;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.MemoryRange;
import org.neo4j.graphalgo.core.utils.mem.MemoryUsage;
import org.neo4j.graphalgo.core.utils.paged.PageUtil;

import static org.neo4j.graphalgo.RelationshipType.ALL_RELATIONSHIPS;
import static org.neo4j.graphalgo.core.loading.VarLongEncoding.encodedVLongSize;
import static org.neo4j.graphalgo.core.utils.BitUtil.ceilDiv;
import static org.neo4j.graphalgo.core.utils.paged.PageUtil.indexInPage;
import static org.neo4j.graphalgo.core.utils.paged.PageUtil.pageIndex;

public final class TransientAdjacencyList implements AdjacencyList {

    public static final int PAGE_SHIFT = 18;
    public static final int PAGE_SIZE = 1 << PAGE_SHIFT;
    public static final long PAGE_MASK = PAGE_SIZE - 1;

    private final long allocatedMemory;
    private byte[][] pages;

    public static MemoryEstimation compressedMemoryEstimation(long avgDegree, long nodeCount) {
        // Best case scenario:
        // Difference between node identifiers in each adjacency list is 1.
        // This leads to ideal compression through delta encoding.
        int deltaBestCase = 1;
        long bestCaseAdjacencySize = computeAdjacencyByteSize(avgDegree, nodeCount, deltaBestCase);

        // Worst case scenario:
        // Relationships are equally distributed across nodes, i.e. each node has the same number of rels.
        // Within each adjacency list, all identifiers have the highest possible difference between each other.
        // Highest possible difference is the number of nodes divided by the average degree.
        long deltaWorstCase = (avgDegree > 0) ? ceilDiv(nodeCount, avgDegree) : 0L;
        long worstCaseAdjacencySize = computeAdjacencyByteSize(avgDegree, nodeCount, deltaWorstCase);

        int minPages = PageUtil.numPagesFor(bestCaseAdjacencySize, PAGE_SHIFT, PAGE_MASK);
        int maxPages = PageUtil.numPagesFor(worstCaseAdjacencySize, PAGE_SHIFT, PAGE_MASK);

        long bytesPerPage = MemoryUsage.sizeOfByteArray(PAGE_SIZE);
        long minMemoryReqs = minPages * bytesPerPage + MemoryUsage.sizeOfObjectArray(minPages);
        long maxMemoryReqs = maxPages * bytesPerPage + MemoryUsage.sizeOfObjectArray(maxPages);

        MemoryRange pagesMemoryRange = MemoryRange.of(minMemoryReqs, maxMemoryReqs);

        return MemoryEstimations
            .builder(TransientAdjacencyList.class)
            .fixed("pages", pagesMemoryRange)
            .build();
    }

    public static MemoryEstimation compressedMemoryEstimation(boolean undirected) {
        return compressedMemoryEstimation(ALL_RELATIONSHIPS, undirected);
    }

    public static MemoryEstimation compressedMemoryEstimation(RelationshipType relationshipType, boolean undirected) {
        return MemoryEstimations.setup("", dimensions -> {
            long nodeCount = dimensions.nodeCount();
            long relCountForType = dimensions.relationshipCounts().getOrDefault(relationshipType, dimensions.maxRelCount());
            long relCount = undirected ? relCountForType * 2 : relCountForType;
            long avgDegree = (nodeCount > 0) ? ceilDiv(relCount, nodeCount) : 0L;
            return TransientAdjacencyList.compressedMemoryEstimation(avgDegree, nodeCount);
        });
    }

    public static MemoryEstimation uncompressedMemoryEstimation(boolean undirected) {
        return uncompressedMemoryEstimation(ALL_RELATIONSHIPS, undirected);
    }

    public static MemoryEstimation uncompressedMemoryEstimation(RelationshipType relationshipType, boolean undirected) {

        return MemoryEstimations
            .builder(TransientAdjacencyList.class)
            .perGraphDimension("pages", (dimensions, concurrency) -> {
                long nodeCount = dimensions.nodeCount();
                long relCountForType = dimensions.relationshipCounts().getOrDefault(relationshipType, dimensions.maxRelCount());
                long relCount = undirected ? relCountForType * 2 : relCountForType;

                long uncompressedAdjacencySize = relCount * Long.BYTES + nodeCount * Integer.BYTES;
                int pages = PageUtil.numPagesFor(uncompressedAdjacencySize, PAGE_SHIFT, PAGE_MASK);
                long bytesPerPage = MemoryUsage.sizeOfByteArray(PAGE_SIZE);

                return MemoryRange.of(pages * bytesPerPage + MemoryUsage.sizeOfObjectArray(pages));
            })
            .build();
    }

    /* test private */
    static long computeAdjacencyByteSize(long avgDegree, long nodeCount, long delta) {
        long firstAdjacencyIdAvgByteSize = (avgDegree > 0) ? ceilDiv(encodedVLongSize(nodeCount), 2) : 0L;
        int relationshipByteSize = encodedVLongSize(delta);
        int degreeByteSize = Integer.BYTES;
        long compressedAdjacencyByteSize = relationshipByteSize * Math.max(0, (avgDegree - 1));
        return (degreeByteSize + firstAdjacencyIdAvgByteSize + compressedAdjacencyByteSize) * nodeCount;
    }

    public TransientAdjacencyList(byte[][] pages) {
        this.pages = pages;
        this.allocatedMemory = memoryOfPages(pages);
    }

    private static long memoryOfPages(byte[][] pages) {
        long memory = MemoryUsage.sizeOfObjectArray(pages.length);
        for (byte[] page : pages) {
            if (page != null) {
                memory += MemoryUsage.sizeOfByteArray(page.length);
            }
        }
        return memory;
    }

    @Override
    public int degree(long index) {
        return AdjacencyDecompressingReader.readInt(
                pages[pageIndex(index, PAGE_SHIFT)],
                indexInPage(index, PAGE_MASK));
    }

    @Override
    public void close() {
        pages = null;
    }

    // Cursors

    @Override
    public Cursor cursor(long offset) {
        return new Cursor(pages).init(offset);
    }

    @Override
    public DecompressingCursor rawDecompressingCursor() {
        return new DecompressingCursor(pages);
    }

    @Override
    public DecompressingCursor decompressingCursor(long offset) {
        return rawDecompressingCursor().init(offset);
    }

    /**
     * Initialise the given cursor with the given offset
     */
    static DecompressingCursor decompressingCursor(DecompressingCursor reuse, long offset) {
        return reuse.init(offset);
    }

    public static final class Cursor extends MutableIntValue implements PropertyCursor {

        static final Cursor EMPTY = new Cursor(new byte[0][]);

        private byte[][] pages;

        private byte[] currentPage;
        private int degree;
        private int offset;
        private int limit;

        private Cursor(byte[][] pages) {
            this.pages = pages;
        }

        public int length() {
            return degree;
        }

        @Override
        public boolean hasNextLong() {
            return offset < limit;
        }

        @Override
        public long nextLong() {
            long value = AdjacencyDecompressingReader.readLong(currentPage, offset);
            offset += Long.BYTES;
            return value;
        }

        Cursor init(long fromIndex) {
            this.currentPage = pages[pageIndex(fromIndex, PAGE_SHIFT)];
            this.offset = indexInPage(fromIndex, PAGE_MASK);
            this.degree = AdjacencyDecompressingReader.readInt(currentPage, offset);
            this.offset += Integer.BYTES;
            this.limit = offset + degree * Long.BYTES;
            return this;
        }

        @Override
        public void close() {
            pages = null;
        }
    }

    public static final class DecompressingCursor extends MutableIntValue implements AdjacencyCursor {

        static final long NOT_FOUND = -1;
        private byte[][] pages;
        private final AdjacencyDecompressingReader decompress;

        private int maxTargets;
        private int currentPosition;

        private DecompressingCursor(byte[][] pages) {
            this.pages = pages;
            this.decompress = new AdjacencyDecompressingReader();
        }

        DecompressingCursor init(long fromIndex) {
            maxTargets = decompress.reset(
                pages[pageIndex(fromIndex, PAGE_SHIFT)],
                indexInPage(fromIndex, PAGE_MASK));
            currentPosition = 0;
            return this;
        }

        /**
         * Copy iteration state from another cursor without changing {@code other}.
         */
        void copyFrom(DecompressingCursor other) {
            decompress.copyFrom(other.decompress);
            currentPosition = other.currentPosition;
            maxTargets = other.maxTargets;
        }

        @Override
        public int size() {
            return maxTargets;
        }

        @Override
        public int remaining() {
            return maxTargets - currentPosition;
        }

        @Override
        public boolean hasNextVLong() {
            return currentPosition < maxTargets;
        }

        @Override
        public long nextVLong() {
            int current = currentPosition++;
            int remaining = maxTargets - current;
            return decompress.next(remaining);
        }

        @Override
        public long peekVLong() {
            int remaining = maxTargets - currentPosition;
            return decompress.peek(remaining);
        }

        // TODO: I think this documentation if either out of date or misleading.
        //  Either we skip all blocks and return -1 or we find a value that is strictly larger.
        /**
         * Read and decode target ids until it is strictly larger than ({@literal >}) the provided {@code target}.
         * Might return an id that is less than or equal to {@code target} iff the cursor did exhaust before finding an
         * id that is large enough.
         * {@code skipUntil(target) <= target} can be used to distinguish the no-more-ids case and afterwards {@link #hasNextVLong()}
         * will return {@code false}
         */
        long skipUntil(long target) {
            long value = decompress.skipUntil(target, remaining(), this);
            this.currentPosition += this.value;
            return value;
        }

        /**
         * Read and decode target ids until it is larger than or equal ({@literal >=}) the provided {@code target}.
         * Might return an id that is less than {@code target} iff the cursor did exhaust before finding an
         * id that is large enough.
         * {@code advance(target) < target} can be used to distinguish the no-more-ids case and afterwards {@link #hasNextVLong()}
         * will return {@code false}
         */
        long advance(long target) {
            int targetsLeftToBeDecoded = remaining();
            if(targetsLeftToBeDecoded <= 0) {
                return NOT_FOUND;
            }
            long value = decompress.advance(target, targetsLeftToBeDecoded, this);
            this.currentPosition += this.value;
            return value;
        }

        @Override
        public void close() {
            pages = null;
        }
    }
}
