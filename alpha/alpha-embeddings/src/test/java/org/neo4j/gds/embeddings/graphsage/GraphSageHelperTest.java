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
package org.neo4j.gds.embeddings.graphsage;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.beta.generator.PropertyProducer;
import org.neo4j.graphalgo.beta.generator.RandomGraphGenerator;
import org.neo4j.graphalgo.beta.generator.RelationshipDistribution;
import org.neo4j.graphalgo.config.RandomGraphGeneratorConfig;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class GraphSageHelperTest {

    @Test
    void shouldInitializeFeaturesCorrectly() {
        RandomGraphGenerator randomGraphGenerator = RandomGraphGenerator.builder()
            .nodeCount(20)
            .averageDegree(3)
            .nodePropertyProducer(PropertyProducer.fixed("dummyProperty", 5D))
            .relationshipDistribution(RelationshipDistribution.POWER_LAW)
            .seed(123L)
            .aggregation(Aggregation.SINGLE)
            .orientation(Orientation.UNDIRECTED)
            .allowSelfLoops(RandomGraphGeneratorConfig.AllowSelfLoops.NO)
            .allocationTracker(AllocationTracker.empty())
            .build();
        var graph = randomGraphGenerator.generate();
        HugeObjectArray<double[]> properties = GraphSageHelper.initializeFeatures(
            graph,
            List.of("dummyProperty"),
            false
        );

        assertNotNull(properties);
        for(int i = 0; i < properties.size(); i++) {
            double[] doubles = properties.get(i);
            assertArrayEquals(new double[] { 5D }, doubles );
        }
    }
}
