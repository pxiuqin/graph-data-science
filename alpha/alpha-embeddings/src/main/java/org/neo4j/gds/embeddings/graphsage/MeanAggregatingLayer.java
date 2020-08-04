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

import org.neo4j.gds.embeddings.graphsage.ddl4j.Variable;
import org.neo4j.gds.embeddings.graphsage.ddl4j.functions.Weights;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

public class MeanAggregatingLayer implements Layer {

    private final UniformNeighborhoodSampler sampler;
    private final long sampleSize;
    private final Weights weights;
    private long randomState;
    private final Function<Variable, Variable> activationFunction;

    public MeanAggregatingLayer(Weights weights, long sampleSize, Function<Variable, Variable> activationFunction) {
        this.sampleSize = sampleSize;
        this.weights = weights;
        this.activationFunction = activationFunction;
        this.randomState = ThreadLocalRandom.current().nextLong();
        this.sampler = new UniformNeighborhoodSampler();
    }

    @Override
    public Aggregator aggregator() {
        return new MeanAggregator(weights, activationFunction);
    }

    @Override
    public UniformNeighborhoodSampler sampler() {
        return sampler;
    }

    @Override
    public long sampleSize() {
        return sampleSize;
    }

    @Override
    public long randomState() {
        return randomState;
    }

    @Override
    public void generateNewRandomState() {
        randomState = ThreadLocalRandom.current().nextLong();
    }
}
