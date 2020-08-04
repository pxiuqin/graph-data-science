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
package org.neo4j.gds.embeddings.node2vec;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.TestProgressLogger;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.params.provider.Arguments.arguments;

class PositiveSampleProducerTest {

    private final long[] buffer = new long[2];
    private final HugeDoubleArray centerNodeProbabilities = HugeDoubleArray.of(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    );

    @ParameterizedTest(name = "{0}")
    @MethodSource("org.neo4j.gds.embeddings.node2vec.PositiveSampleProducerTest#pairCombinations")
    void shouldProducePairsWith(
        String name,
        int windowSize,
        HugeObjectArray<long[]> walks,
        List<Pair<Long, Long>> expectedPairs
    ) {
        Collection<Pair<Long, Long>> actualPairs = new ArrayList<>();

        PositiveSampleProducer producer = new PositiveSampleProducer(
            walks,
            centerNodeProbabilities,
            0,
            walks.size() - 1L,
            windowSize,
            TestProgressLogger.NULL_LOGGER
        );
        while (producer.hasNext()) {
            producer.next(buffer);
            actualPairs.add(Pair.of(buffer[0], buffer[1]));
        }

        assertEquals(expectedPairs, actualPairs);
    }

    @Test
    void shouldProducePairsWithBounds() {
        HugeObjectArray<long[]> walks = HugeObjectArray.of(
            new long[]{0, 1, 2},
            new long[]{3, 4, 5},
            new long[]{3, 4, 5},
            new long[]{3, 4, 5}
        );

        Collection<Pair<Long, Long>> actualPairs = new ArrayList<>();
        PositiveSampleProducer producer = new PositiveSampleProducer(
            walks,
            centerNodeProbabilities,
            0,
            1,
            3,
            TestProgressLogger.NULL_LOGGER
        );
        while (producer.hasNext()) {
            producer.next(buffer);
            actualPairs.add(Pair.of(buffer[0], buffer[1]));
        }

        assertEquals(
            List.of(
                Pair.of(0L, 1L),
                Pair.of(1L, 0L),
                Pair.of(1L, 2L),
                Pair.of(2L, 1L),

                Pair.of(3L, 4L),
                Pair.of(4L, 3L),
                Pair.of(4L, 5L),
                Pair.of(5L, 4L)
            ),
            actualPairs
        );
    }

    @Test
    void shouldRemoveDownsampledWordFromWalk() {
        HugeObjectArray<long[]> walks = HugeObjectArray.of(
            new long[]{0, 1},       // 1 is downsampled, and the walk is then too short and will be ignored
            new long[]{0, 1, 2},    // 1 is downsampled, the remaining walk is (0,2)
            new long[]{3, 4, 5, 6}, // 5 is downsampled, the remaining walk is (3,4,6)
            new long[]{3, 4, 5}     // 5 is downsampled, the remaining walk is (3,4)
        );

        HugeDoubleArray centerNodeProbabilities = HugeDoubleArray.of(
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0
        );

        Collection<Pair<Long, Long>> actualPairs = new ArrayList<>();
        PositiveSampleProducer producer = new PositiveSampleProducer(
            walks,
            centerNodeProbabilities,
            0,
            3,
            3,
            TestProgressLogger.NULL_LOGGER
        );

        while (producer.hasNext()) {
            producer.next(buffer);
            actualPairs.add(Pair.of(buffer[0], buffer[1]));
        }

        assertEquals(
            List.of(
                Pair.of(0L, 2L),
                Pair.of(2L, 0L),

                Pair.of(3L, 4L),
                Pair.of(4L, 3L),
                Pair.of(4L, 6L),
                Pair.of(6L, 4L),

                Pair.of(3L, 4L),
                Pair.of(4L, 3L)
            ),
            actualPairs
        );
    }

    static Stream<Arguments> pairCombinations() {
        return Stream.of(
            arguments(
                "Uneven window size",
                3,
                HugeObjectArray.of(
                    new long[]{0, 1, 2}
                ),
                List.of(
                    Pair.of(0L, 1L),
                    Pair.of(1L, 0L),
                    Pair.of(1L, 2L),
                    Pair.of(2L, 1L)
                )
            ),

            arguments(
                "Even window size",
                4,
                HugeObjectArray.of(
                    new long[]{0, 1, 2, 3}
                ),
                List.of(
                    Pair.of(0L, 1L),
                    Pair.of(1L, 0L),
                    Pair.of(1L, 2L),
                    Pair.of(2L, 0L),
                    Pair.of(2L, 1L),
                    Pair.of(2L, 3L),
                    Pair.of(3L, 1L),
                    Pair.of(3L, 2L)
                )
            ),

            arguments(
                "Window size greater than walk length",
                3,
                HugeObjectArray.of(
                    new long[]{0, 1}
                ),
                List.of(
                    Pair.of(0L, 1L),
                    Pair.of(1L, 0L)
                )
            ),

            arguments(
                "Multiple walks",
                3,
                HugeObjectArray.of(
                    new long[]{0, 1, 2},
                    new long[]{3, 4, 5}
                ),
                List.of(
                    Pair.of(0L, 1L),
                    Pair.of(1L, 0L),
                    Pair.of(1L, 2L),
                    Pair.of(2L, 1L),

                    Pair.of(3L, 4L),
                    Pair.of(4L, 3L),
                    Pair.of(4L, 5L),
                    Pair.of(5L, 4L)
                )
            )
        );
    }

}
