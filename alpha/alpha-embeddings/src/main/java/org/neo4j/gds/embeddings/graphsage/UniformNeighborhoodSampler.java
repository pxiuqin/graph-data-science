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

import org.neo4j.graphalgo.api.Graph;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

public class UniformNeighborhoodSampler {
    private final Random random;

    public UniformNeighborhoodSampler() {
        this.random = new Random();
    }

    public List<Long> sample(Graph graph, long nodeId, long numberOfSamples, long randomState) {
        AtomicLong remainingToSample = new AtomicLong(numberOfSamples);
        AtomicLong remainingToConsider = new AtomicLong(graph.degree(nodeId));
        List<Long> neighbors = new ArrayList<>();
        graph.concurrentCopy().forEachRelationship(
            nodeId,
            (source, target) -> {
                if (remainingToSample.get() == 0 || remainingToConsider.get() == 0) {
                    return false;
                }
                double randomDouble = randomDouble(randomState, source, target, graph.nodeCount());
                if (remainingToConsider.getAndDecrement() * randomDouble <= remainingToSample.get()) {
                    neighbors.add(target);
                    remainingToSample.decrementAndGet();
                }
                return true;
            }
        );
        return neighbors;
    }

    private double randomDouble(long randomState, long source, long target, long nodeCount) {
        random.setSeed(randomState + source + nodeCount * target);
        return random.nextDouble();
    }
}
