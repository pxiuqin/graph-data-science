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
import org.neo4j.graphalgo.api.IntersectionConsumer;
import org.neo4j.graphalgo.api.RelationshipIntersect;

import java.util.function.LongPredicate;

/**
 * An instance of this is not thread-safe; Iteration/Intersection on multiple threads will
 * throw misleading {@link NullPointerException}s.
 * Instances are however safe to use concurrently with other {@link org.neo4j.graphalgo.api.RelationshipIterator}s.
 */

public abstract class GraphIntersect<CURSOR extends AdjacencyCursor> implements RelationshipIntersect {

    protected CURSOR empty;
    private final CURSOR cache;
    private final CURSOR cacheA;
    private final CURSOR cacheB;
    private final LongPredicate degreeFilter;

    protected GraphIntersect(
        CURSOR cache,
        CURSOR cacheA,
        CURSOR cacheB,
        CURSOR empty,
        long maxDegree
    ) {
        this.cache = cache;
        this.cacheA = cacheA;
        this.cacheB = cacheB;
        this.empty = empty;

        this.degreeFilter = maxDegree < Long.MAX_VALUE
            ? (node) -> degree(node) <= maxDegree
            : (ignore) -> true;
    }

    @Override
    public void intersectAll(long nodeIdA, IntersectionConsumer consumer) {
        if(!degreeFilter.test(nodeIdA)) {
            return;
        }

        CURSOR mainDecompressingCursor = cursor(nodeIdA, cache);
        long nodeIdB = skipUntil(mainDecompressingCursor, nodeIdA);
        if (nodeIdB <= nodeIdA) {
            return;
        }

        CURSOR lead;
        CURSOR follow;
        CURSOR decompressingCursorA = cacheA;
        CURSOR decompressingCursorB = cacheB;

        long nodeIdC;
        long currentA;
        long s;
        long t;

        long lastNodeB;
        long lastNodeC;
        while (mainDecompressingCursor.hasNextVLong()) {
            lastNodeC = -1;
            if (degreeFilter.test(nodeIdB)) {
                decompressingCursorB = cursor(nodeIdB, decompressingCursorB);
                nodeIdC = skipUntil(decompressingCursorB, nodeIdB);
                if (nodeIdC > nodeIdB && degreeFilter.test(nodeIdC)) {
                    copyFrom(mainDecompressingCursor, decompressingCursorA);
                    currentA = advance(decompressingCursorA, nodeIdC);

                    if (currentA == nodeIdC && nodeIdC > lastNodeC) {
                        consumer.accept(nodeIdA, nodeIdB, nodeIdC);
                        lastNodeC = nodeIdC;
                    }

                    if (decompressingCursorA.remaining() <= decompressingCursorB.remaining()) {
                        lead = decompressingCursorA;
                        follow = decompressingCursorB;
                    } else {
                        lead = decompressingCursorB;
                        follow = decompressingCursorA;
                    }

                    while (lead.hasNextVLong() && follow.hasNextVLong()) {
                        s = lead.nextVLong();
                        t = advance(follow, s);
                        if (t == s && t > lastNodeC) {
                            consumer.accept(nodeIdA, nodeIdB, s);
                            lastNodeC = t;
                        }
                    }
                }
            }

            lastNodeB = nodeIdB;
            while (mainDecompressingCursor.hasNextVLong() && nodeIdB == lastNodeB) {
                nodeIdB = mainDecompressingCursor.nextVLong();
            }
        }
    }

    protected abstract long skipUntil(CURSOR cursor, long nodeId);

    protected abstract long advance(CURSOR cursor, long nodeId);

    protected abstract void copyFrom(CURSOR sourceCursor, CURSOR targetCursor);

    protected abstract CURSOR cursor(long node, CURSOR reuse);

    protected abstract int degree(long node);
}
