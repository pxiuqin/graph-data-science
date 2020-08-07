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
package org.neo4j.graphalgo.beta.generator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.TestSupport;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.RandomGraphGeneratorConfig.AllowSelfLoops;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.huge.HugeGraph;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RandomGraphGeneratorTest {

    @Test
    void shouldGenerateRelsUniformDistributed() {
        int nbrNodes = 10;
        long avgDeg = 5L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        Assertions.assertEquals(graph.nodeCount(), nbrNodes);
        Assertions.assertEquals(nbrNodes * avgDeg, graph.relationshipCount());

        graph.forEachNode((nodeId) -> {
            long[] degree = {0L};

            graph.forEachRelationship(nodeId, (a, b) -> {
                degree[0] = degree[0] + 1;
                return true;
            });

            Assertions.assertEquals(avgDeg, degree[0]);
            return true;
        });
    }

    @Test
    void shouldGenerateRelsPowerLawDistributed() {
        int nbrNodes = 10000;
        long avgDeg = 5L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.POWER_LAW)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        Assertions.assertEquals(graph.nodeCount(), nbrNodes);
        Assertions.assertEquals((double) nbrNodes * avgDeg, graph.relationshipCount(), 1000D);
    }

    @Test
    void shouldNotGenerateSelfLoops() {
        int nbrNodes = 1000;
        long avgDeg = 5L;
        AllowSelfLoops allowSelfLoops = AllowSelfLoops.NO;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.POWER_LAW)
            .aggregation(Aggregation.NONE)
            .orientation(Orientation.UNDIRECTED)
            .allowSelfLoops(allowSelfLoops)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        for (long nodeId = 0; nodeId < graph.nodeCount(); nodeId++) {
            graph.forEachRelationship(nodeId, (src, trg) -> {
               Assertions.assertNotEquals(src, trg);
               return true;
            });
        }
    }

    @Test
    void shouldGenerateRelsRandomDistributed() {
        int nbrNodes = 1000;
        long avgDeg = 5L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.RANDOM)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        Assertions.assertEquals(graph.nodeCount(), nbrNodes);

        List<Long> degrees = new ArrayList<Long>();
        graph.forEachNode((nodeId) -> {
            long[] degree = {0L};

            graph.forEachRelationship(nodeId, (a, b) -> {
                degree[0] = degree[0] + 1;
                return true;
            });

            degrees.add(degree[0]);
            return true;
        });

        double actualAverage = degrees.stream().reduce(Long::sum).orElseGet(() -> 0L) / (double) degrees.size();
        Assertions.assertEquals((double) avgDeg, actualAverage, 1D);
    }

    @Test
    void shouldGenerateRelationshipPropertiesWithFixedValue() {
        int nbrNodes = 10;
        long avgDeg = 5L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .relationshipPropertyProducer(PropertyProducer.fixed("property", 42D))
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        graph.forEachNode((nodeId) -> {
            graph.forEachRelationship(nodeId, Double.NaN, (s, t, p) -> {
                Assertions.assertEquals(42D, p);
                return true;
            });
            return true;
        });
    }

    @Test
    void shouldGenerateRelationshipWithRandom() {
        int nbrNodes = 10;
        long avgDeg = 5L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .relationshipPropertyProducer(PropertyProducer.random("prop", -10, 10))
            .allocationTracker(AllocationTracker.EMPTY)
            .build();
        HugeGraph graph = randomGraphGenerator.generate();

        graph.forEachNode((nodeId) -> {
            graph.forEachRelationship(nodeId, Double.NaN, (s, t, p) -> {
                assertTrue(p >= -10);
                assertTrue(p <= 10);
                return true;
            });
            return true;
        });
    }

    @Test
    void shouldGenerateNodeProperties() {
        HugeGraph graph = RandomGraphGenerator.builder()
            .nodeCount(10)
            .averageDegree(2)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .allocationTracker(AllocationTracker.EMPTY)
            .nodePropertyProducer(PropertyProducer.random("foo", 0, 1))
            .build()
            .generate();

        NodeProperties nodeProperties = graph.nodeProperties("foo");
        for (int nodeId = 0; nodeId < 10; nodeId++) {
            double value = nodeProperties.getDouble(nodeId);
            assertTrue(0 <= value && value <= 1);
        }
    }

    @Test
    void shouldBeSeedAble() {
        int nbrNodes = 10;
        long avgDeg = 5L;
        long seed = 1337L;

        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .seed(seed)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();

        RandomGraphGenerator otherRandomGenerator = RandomGraphGenerator.builder()
            .nodeCount(nbrNodes)
            .averageDegree(avgDeg)
            .relationshipDistribution(RelationshipDistribution.UNIFORM)
            .seed(seed)
            .allocationTracker(AllocationTracker.EMPTY)
            .build();

        HugeGraph graph1 = randomGraphGenerator.generate();
        HugeGraph graph2 = otherRandomGenerator.generate();

        TestSupport.assertGraphEquals(graph1, graph2);
    }
}
