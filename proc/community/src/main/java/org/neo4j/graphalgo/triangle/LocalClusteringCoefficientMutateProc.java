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

import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.MutateProc;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.graphalgo.results.MemoryEstimateResult;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.triangle.LocalClusteringCoefficientCompanion.warnOnGraphWithParallelRelationships;
import static org.neo4j.procedure.Mode.READ;
import static org.neo4j.procedure.Mode.WRITE;

public class LocalClusteringCoefficientMutateProc extends MutateProc<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientMutateProc.MutateResult, LocalClusteringCoefficientMutateConfig> {

    @Procedure(value = "gds.localClusteringCoefficient.mutate", mode = WRITE)
    @Description("")
    public Stream<MutateResult> write(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return mutate(compute(graphNameOrConfig, configuration));
    }

    @Procedure(value = "gds.localClusteringCoefficient.mutate.estimate", mode = READ)
    @Description(ESTIMATE_DESCRIPTION)
    public Stream<MemoryEstimateResult> estimate(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return computeEstimate(graphNameOrConfig, configuration);
    }

    @Override
    protected LocalClusteringCoefficientMutateConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return LocalClusteringCoefficientMutateConfig.of(
            username,
            graphName,
            maybeImplicitCreate,
            config
        );
    }

    @Override
    protected AlgorithmFactory<LocalClusteringCoefficient, LocalClusteringCoefficientMutateConfig> algorithmFactory() {
        return new LocalClusteringCoefficientFactory<>();
    }

    @Override
    protected void validateConfigs(
        GraphCreateConfig graphCreateConfig, LocalClusteringCoefficientMutateConfig config
    ) {
        validateIsUndirectedGraph(graphCreateConfig, config);
        warnOnGraphWithParallelRelationships(graphCreateConfig, config, log);
    }

    @Override
    protected NodeProperties getNodeProperties(
        ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientMutateConfig> computationResult
    ) {
        return LocalClusteringCoefficientCompanion.nodeProperties(computationResult);
    }

    @Override
    protected AbstractResultBuilder<MutateResult> resultBuilder(ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientMutateConfig> computeResult) {
        return LocalClusteringCoefficientCompanion.resultBuilder(
            new LocalClusteringCoefficientMutateBuilder(callContext, computeResult.tracker()),
            computeResult
        );
    }

    public static class MutateResult extends LocalClusteringCoefficientStatsProc.StatsResult {

        public long nodePropertiesWritten;
        public long mutateMillis;

        public MutateResult(
            double averageClusteringCoefficient,
            long nodeCount,
            long nodePropertiesWritten,
            long createMillis,
            long computeMillis,
            long mutateMillis,
            Map<String, Object> configuration
        ) {
            super(
                averageClusteringCoefficient,
                nodeCount,
                createMillis,
                computeMillis,
                configuration
            );
            this.nodePropertiesWritten = nodePropertiesWritten;
            this.mutateMillis = mutateMillis;
        }
    }

    static class LocalClusteringCoefficientMutateBuilder extends LocalClusteringCoefficientCompanion.ResultBuilder<MutateResult> {

        LocalClusteringCoefficientMutateBuilder(
            ProcedureCallContext callContext,
            AllocationTracker tracker
        ) {
            super(callContext, tracker);
        }

        @Override
        protected MutateResult buildResult() {
            return new MutateResult(
                averageClusteringCoefficient,
                nodeCount,
                nodePropertiesWritten,
                createMillis,
                computeMillis,
                mutateMillis,
                config.toMap()
            );
        }
    }
}
