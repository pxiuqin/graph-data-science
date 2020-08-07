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

import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.config.RandomGraphGeneratorConfig;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;

import java.util.Optional;

public class RandomGraphGeneratorBuilder {
    private long nodeCount;
    private long averageDegree;
    private RelationshipDistribution relationshipDistribution;
    private long seed = 0L;
    private Optional<PropertyProducer> maybeNodePropertyProducer = Optional.empty();
    private Optional<PropertyProducer> maybeRelationshipPropertyProducer = Optional.empty();
    private Aggregation aggregation = Aggregation.NONE;
    private Orientation orientation = Orientation.NATURAL;
    private RandomGraphGeneratorConfig.AllowSelfLoops allowSelfLoops = RandomGraphGeneratorConfig.AllowSelfLoops.NO;
    private AllocationTracker allocationTracker;

    public RandomGraphGeneratorBuilder nodeCount(long nodeCount) {
        this.nodeCount = nodeCount;
        return this;
    }

    public RandomGraphGeneratorBuilder averageDegree(long averageDegree) {
        this.averageDegree = averageDegree;
        return this;
    }

    public RandomGraphGeneratorBuilder relationshipDistribution(RelationshipDistribution relationshipDistribution) {
        this.relationshipDistribution = relationshipDistribution;
        return this;
    }

    public RandomGraphGeneratorBuilder seed(long seed) {
        this.seed = seed;
        return this;
    }

    public RandomGraphGeneratorBuilder nodePropertyProducer(PropertyProducer nodePropertyProducer) {
        this.maybeNodePropertyProducer = Optional.of(nodePropertyProducer);
        return this;
    }

    public RandomGraphGeneratorBuilder relationshipPropertyProducer(PropertyProducer relationshipPropertyProducer) {
        this.maybeRelationshipPropertyProducer = Optional.of(relationshipPropertyProducer);
        return this;
    }

    public RandomGraphGeneratorBuilder aggregation(Aggregation aggregation) {
        this.aggregation = aggregation;
        return this;
    }

    public RandomGraphGeneratorBuilder orientation(Orientation orientation) {
        this.orientation = orientation;
        return this;
    }

    public RandomGraphGeneratorBuilder allowSelfLoops(RandomGraphGeneratorConfig.AllowSelfLoops allowSelfLoops) {
        this.allowSelfLoops = allowSelfLoops;
        return this;
    }

    public RandomGraphGeneratorBuilder allocationTracker(AllocationTracker allocationTracker) {
        this.allocationTracker = allocationTracker;
        return this;
    }

    public RandomGraphGenerator build() {
        validate();
        return new RandomGraphGenerator(
            nodeCount,
            averageDegree,
            relationshipDistribution,
            seed,
            maybeNodePropertyProducer,
            maybeRelationshipPropertyProducer,
            aggregation,
            orientation,
            allowSelfLoops,
            allocationTracker
        );
    }

    private void validate() {
        if (nodeCount <= 0) {
            throw new IllegalArgumentException("Must provide positive nodeCount");
        }
        if (averageDegree <= 0) {
            throw new IllegalArgumentException("Must provide positive averageDegree");
        }
        if (relationshipDistribution == null) {
            throw new IllegalArgumentException("Must provide a RelationshipDistribution");
        }
        if (allocationTracker == null) {
            throw new IllegalArgumentException("Must provide a AllocationTracker");
        }
    }
}
