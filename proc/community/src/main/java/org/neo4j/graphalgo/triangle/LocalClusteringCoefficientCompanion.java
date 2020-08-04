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
package org.neo4j.graphalgo.triangle;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeDoubleArray;
import org.neo4j.graphalgo.result.AbstractCommunityResultBuilder;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;
import org.neo4j.logging.Log;

import java.util.Collections;
import java.util.Optional;

import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;

final class LocalClusteringCoefficientCompanion {

    private LocalClusteringCoefficientCompanion() {}

    static <CONFIG extends LocalClusteringCoefficientBaseConfig> NodeProperties nodeProperties(
        AlgoBaseProc.ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, CONFIG> computeResult
    ) {
        return computeResult.result().localClusteringCoefficients().asNodeProperties();
    }

    static void warnOnGraphWithParallelRelationships(GraphCreateConfig graphCreateConfig, LocalClusteringCoefficientBaseConfig config, Log log) {
        if (graphCreateConfig instanceof GraphCreateFromStoreConfig) {
            GraphCreateFromStoreConfig storeConfig = (GraphCreateFromStoreConfig) graphCreateConfig;
            storeConfig.relationshipProjections().projections().entrySet().stream()
                .filter(entry -> config.relationshipTypes().equals(Collections.singletonList(PROJECT_ALL)) ||
                                 config.relationshipTypes().contains(entry.getKey().name()))
                .filter(entry -> entry.getValue().isMultiGraph())
                .forEach(entry -> log.warn(
                    "Procedure runs optimal with relationship aggregation." +
                    " Projection for `%s` does not aggregate relationships." +
                    " You might experience a slowdown in the procedure execution.",
                    entry.getKey().equals(RelationshipType.ALL_RELATIONSHIPS) ? "*" : entry.getKey().name
                ));
        }
    }

    static <PROC_RESULT, CONFIG extends LocalClusteringCoefficientBaseConfig> AbstractResultBuilder<PROC_RESULT> resultBuilder(
        ResultBuilder<PROC_RESULT> procResultBuilder,
        AlgoBaseProc.ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, CONFIG> computeResult
    ) {
        var result = Optional.ofNullable(computeResult.result())
            .orElse(EmptyResult.EMPTY_RESULT);

        return procResultBuilder
            .withAverageClusteringCoefficient(result.averageClusteringCoefficient());
    }

    abstract static class ResultBuilder<PROC_RESULT> extends AbstractCommunityResultBuilder<PROC_RESULT> {

        double averageClusteringCoefficient = 0;

        ResultBuilder(ProcedureCallContext callContext, AllocationTracker tracker) {
            super(callContext, tracker);
        }

        ResultBuilder<PROC_RESULT> withAverageClusteringCoefficient(double averageClusteringCoefficient) {
            this.averageClusteringCoefficient = averageClusteringCoefficient;
            return this;
        }
    }


    private static final class EmptyResult implements LocalClusteringCoefficient.Result {

        static final EmptyResult EMPTY_RESULT = new EmptyResult();

        private EmptyResult() {}

        @Override
        public HugeDoubleArray localClusteringCoefficients() {
            return HugeDoubleArray.newArray(0, AllocationTracker.EMPTY);
        }

        @Override
        public double averageClusteringCoefficient() {
            return 0;
        }
    }
}
