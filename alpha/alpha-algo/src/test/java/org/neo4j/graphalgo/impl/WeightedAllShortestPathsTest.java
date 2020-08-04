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
package org.neo4j.graphalgo.impl;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.AlgoTestBase;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.StoreLoaderBuilder;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.graphbuilder.GraphBuilder;
import org.neo4j.graphalgo.impl.msbfs.WeightedAllShortestPaths;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Matchers.anyDouble;
import static org.mockito.Matchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 *     1
 *  (0)->(1)
 * 1 | 1 | 1
 *   v   v
 *  (2)->(3)
 * 1 | 1 | 1
 *   v   v
 *  (4)->(5)
 * 1 | 1 | 1
 *   v   v
 *  (6)->(7)
 * 1 | 1  | 1
 *   v    v
 *  (8)->(9)
 */
class WeightedAllShortestPathsTest extends AlgoTestBase {

    private static final int width = 2, height = 5;

    private static final String PROPERTY = "property";
    private static final String LABEL = "Node";
    private static final String RELATIONSHIP = "REL";

    private Graph graph;

    @BeforeEach
    void setup() {
        GraphBuilder.create(db)
            .setLabel(LABEL)
            .setRelationship(RELATIONSHIP)
            .newGridBuilder()
            .createGrid(width, height)
            .forEachRelInTx(rel -> rel.setProperty(PROPERTY, 1.0))
            .close();

        graph = new StoreLoaderBuilder()
            .api(db)
            .addNodeLabel(LABEL)
            .addRelationshipType(RELATIONSHIP)
            .addRelationshipProperty(PropertyMapping.of(PROPERTY, 1.0))
            .build()
            .graph();
    }

    @Test
    void testResults() {

        final ResultConsumer mock = mock(ResultConsumer.class);

        new WeightedAllShortestPaths(graph, Pools.DEFAULT, 4)
                .compute()
                .forEach(r -> {
                    assertNotEquals(Double.POSITIVE_INFINITY, r.distance);
                    if (r.sourceNodeId == r.targetNodeId) {
                        assertEquals(0.0, r.distance, 0.1);
                    }
                    mock.test(r.sourceNodeId, r.targetNodeId, r.distance);
                });

        verify(mock, times(45)).test(anyLong(), anyLong(), anyDouble());

        verify(mock, times(1)).test(0, 9, 5.0);
        verify(mock, times(1)).test(0, 0, 0.0);

    }

    @Test
    void shouldThrowIfGraphHasNoRelationshipProperty() {
        Graph graph = new StoreLoaderBuilder()
            .api(db)
            .addNodeLabel(LABEL)
            .addRelationshipType(RELATIONSHIP)
            .build()
            .graph();

        UnsupportedOperationException exception = assertThrows(UnsupportedOperationException.class, () -> {
            new WeightedAllShortestPaths(graph, Pools.DEFAULT, 4);
        });

        assertTrue(exception.getMessage().contains("not supported"));
    }

    interface ResultConsumer {

        void test(long source, long target, double distance);
    }
}
