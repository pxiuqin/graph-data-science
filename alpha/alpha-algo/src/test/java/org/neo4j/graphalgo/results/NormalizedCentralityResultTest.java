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
package org.neo4j.graphalgo.results;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.impl.utils.NormalizationFunction;
import org.neo4j.graphalgo.result.CentralityResult;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class NormalizedCentralityResultTest {

    @Test
    void maxNormalization() {
        CentralityResultWithStatistics centralityResult = mock(CentralityResultWithStatistics.class);
        when(centralityResult.score(0)).thenReturn(1.0);
        when(centralityResult.score(1)).thenReturn(2.0);
        when(centralityResult.score(2)).thenReturn(3.0);
        when(centralityResult.score(3)).thenReturn(4.0);
        when(centralityResult.computeMax()).thenReturn(4.0);

        CentralityResult normalizedResult = NormalizationFunction.MAX.apply(centralityResult);

        assertEquals(0.25, normalizedResult.score(0), 0.01);
        assertEquals(0.5, normalizedResult.score(1), 0.01);
        assertEquals(0.75, normalizedResult.score(2), 0.01);
        assertEquals(1.0, normalizedResult.score(3), 0.01);
    }

    @Test
    void noNormalization() {
        CentralityResultWithStatistics centralityResult = mock(CentralityResultWithStatistics.class);
        when(centralityResult.score(0)).thenReturn(1.0);
        when(centralityResult.score(1)).thenReturn(2.0);
        when(centralityResult.score(2)).thenReturn(3.0);
        when(centralityResult.score(3)).thenReturn(4.0);
        when(centralityResult.computeMax()).thenReturn(4.0);

        CentralityResult normalizedResult = NormalizationFunction.NONE.apply(centralityResult);

        assertEquals(1.0, normalizedResult.score(0), 0.01);
        assertEquals(2.0, normalizedResult.score(1), 0.01);
        assertEquals(3.0, normalizedResult.score(2), 0.01);
        assertEquals(4.0, normalizedResult.score(3), 0.01);
    }

    @Test
    void l2Norm() {
        CentralityResultWithStatistics centralityResult = mock(CentralityResultWithStatistics.class);
        when(centralityResult.score(0)).thenReturn(1.0);
        when(centralityResult.score(1)).thenReturn(2.0);
        when(centralityResult.score(2)).thenReturn(3.0);
        when(centralityResult.score(3)).thenReturn(4.0);
        when(centralityResult.computeL2Norm()).thenReturn(4.0);

        CentralityResult normalizedResult = NormalizationFunction.L2NORM.apply(centralityResult);

        assertEquals(0.25, normalizedResult.score(0), 0.01);
        assertEquals(0.5, normalizedResult.score(1), 0.01);
        assertEquals(0.75, normalizedResult.score(2), 0.01);
        assertEquals(1.0, normalizedResult.score(3), 0.01);
    }

    @Test
    void l1Norm() {
        CentralityResultWithStatistics centralityResult = mock(CentralityResultWithStatistics.class);
        when(centralityResult.score(0)).thenReturn(1.0);
        when(centralityResult.score(1)).thenReturn(2.0);
        when(centralityResult.score(2)).thenReturn(3.0);
        when(centralityResult.score(3)).thenReturn(4.0);
        when(centralityResult.computeL1Norm()).thenReturn(4.0);

        CentralityResult normalizedResult = NormalizationFunction.L1NORM.apply(centralityResult);

        assertEquals(0.25, normalizedResult.score(0), 0.01);
        assertEquals(0.5, normalizedResult.score(1), 0.01);
        assertEquals(0.75, normalizedResult.score(2), 0.01);
        assertEquals(1.0, normalizedResult.score(3), 0.01);
    }
}
