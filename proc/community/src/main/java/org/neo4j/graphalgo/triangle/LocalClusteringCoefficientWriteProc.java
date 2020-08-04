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
import org.neo4j.graphalgo.WriteProc;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
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
import static org.neo4j.graphalgo.triangle.TriangleCountCompanion.DESCRIPTION;
import static org.neo4j.procedure.Mode.READ;
import static org.neo4j.procedure.Mode.WRITE;

public class LocalClusteringCoefficientWriteProc extends WriteProc<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientWriteProc.WriteResult, LocalClusteringCoefficientWriteConfig> {

    @Procedure(value = "gds.localClusteringCoefficient.write", mode = WRITE)
    @Description(DESCRIPTION)
    public Stream<WriteResult> write(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return write(compute(graphNameOrConfig, configuration));
    }

    @Procedure(value = "gds.localClusteringCoefficient.write.estimate", mode = READ)
    @Description(ESTIMATE_DESCRIPTION)
    public Stream<MemoryEstimateResult> estimateStats(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return computeEstimate(graphNameOrConfig, configuration);
    }

    @Override
    protected LocalClusteringCoefficientWriteConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return LocalClusteringCoefficientWriteConfig.of(
            username,
            graphName,
            maybeImplicitCreate,
            config
        );
    }

    @Override
    protected AlgorithmFactory<LocalClusteringCoefficient, LocalClusteringCoefficientWriteConfig> algorithmFactory() {
        return new LocalClusteringCoefficientFactory<>();
    }

    @Override
    protected void validateConfigs(
        GraphCreateConfig graphCreateConfig, LocalClusteringCoefficientWriteConfig config
    ) {
        validateIsUndirectedGraph(graphCreateConfig, config);
        warnOnGraphWithParallelRelationships(graphCreateConfig, config, log);
    }

    @Override
    protected NodeProperties getNodeProperties(
        ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientWriteConfig> computationResult
    ) {
        return LocalClusteringCoefficientCompanion.nodeProperties(computationResult);
    }

    @Override
    protected AbstractResultBuilder<WriteResult> resultBuilder(ComputationResult<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientWriteConfig> computeResult) {
        return LocalClusteringCoefficientCompanion.resultBuilder(
            new LocalClusteringCoefficientWriteResultBuilder(callContext, computeResult.tracker()),
            computeResult
        );
    }

    public static class WriteResult extends LocalClusteringCoefficientStatsProc.StatsResult {

        public long nodePropertiesWritten;
        public long writeMillis;

        public WriteResult(
            double averageClusteringCoefficient,
            long nodeCount,
            long createMillis,
            long computeMillis,
            long nodePropertiesWritten,
            long writeMillis,
            Map<String, Object> configuration
        ) {
            super(averageClusteringCoefficient, nodeCount, createMillis, computeMillis, configuration);

            this.nodePropertiesWritten = nodePropertiesWritten;
            this.writeMillis = writeMillis;
        }
    }


    static class LocalClusteringCoefficientWriteResultBuilder extends LocalClusteringCoefficientCompanion.ResultBuilder<WriteResult> {

        LocalClusteringCoefficientWriteResultBuilder(
            ProcedureCallContext callContext,
            AllocationTracker tracker
        ) {
            super(callContext, tracker);
        }

        @Override
        protected WriteResult buildResult() {
            return new WriteResult(
                averageClusteringCoefficient,
                nodeCount,
                createMillis,
                computeMillis,
                nodePropertiesWritten,
                writeMillis,
                config.toMap()
            );
        }
    }
}
