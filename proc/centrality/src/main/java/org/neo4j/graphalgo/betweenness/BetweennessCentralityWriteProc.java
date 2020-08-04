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
package org.neo4j.graphalgo.betweenness;

import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.WriteProc;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicDoubleArray;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.graphalgo.results.MemoryEstimateResult;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.betweenness.BetweennessCentralityProc.BETWEENNESS_DESCRIPTION;
import static org.neo4j.procedure.Mode.READ;
import static org.neo4j.procedure.Mode.WRITE;

public class BetweennessCentralityWriteProc extends WriteProc<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityWriteProc.WriteResult, BetweennessCentralityWriteConfig> {

    @Procedure(value = "gds.betweenness.write", mode = WRITE)
    @Description(BETWEENNESS_DESCRIPTION)
    public Stream<WriteResult> write(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return write(compute(graphNameOrConfig, configuration));
    }

    @Procedure(value = "gds.betweenness.write.estimate", mode = READ)
    @Description(BETWEENNESS_DESCRIPTION)
    public Stream<MemoryEstimateResult> estimate(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return computeEstimate(graphNameOrConfig, configuration);
    }

    @Override
    protected BetweennessCentralityWriteConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return BetweennessCentralityWriteConfig.of(username, graphName, maybeImplicitCreate, config);
    }

    @Override
    protected void validateConfigs(GraphCreateConfig graphCreateConfig, BetweennessCentralityWriteConfig config) {
        validateOrientationCombinations(graphCreateConfig, config);
    }

    @Override
    protected AlgorithmFactory<BetweennessCentrality, BetweennessCentralityWriteConfig> algorithmFactory() {
        return BetweennessCentralityProc.algorithmFactory();
    }

    @Override
    protected NodeProperties getNodeProperties(ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityWriteConfig> computationResult) {
        return BetweennessCentralityProc.nodeProperties(computationResult);
    }

    @Override
    protected AbstractResultBuilder<WriteResult> resultBuilder(ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityWriteConfig> computeResult) {
        return BetweennessCentralityProc.resultBuilder(new WriteResult.Builder(), computeResult, callContext);
    }

    public static final class WriteResult extends BetweennessCentralityStatsProc.StatsResult {

        public final long nodePropertiesWritten;
        public final long writeMillis;

        WriteResult(
            long nodePropertiesWritten,
            long createMillis,
            long computeMillis,
            long postProcessingMillis,
            long writeMillis,
            double minCentrality,
            double maxCentrality,
            double sumCentrality,
            Map<String, Object> config
        ) {
            super(
                createMillis,
                computeMillis,
                postProcessingMillis,
                minCentrality,
                maxCentrality,
                sumCentrality,
                config
            );
            this.nodePropertiesWritten = nodePropertiesWritten;
            this.writeMillis = writeMillis;
        }

        static final class Builder extends BetweennessCentralityProc.BetweennessCentralityResultBuilder<WriteResult> {

            @Override
            public WriteResult build() {
                return new WriteResult(
                    nodePropertiesWritten,
                    createMillis,
                    computeMillis,
                    postProcessingMillis,
                    writeMillis,
                    minimumScore,
                    maximumScore,
                    scoreSum,
                    config.toMap()
                );
            }
        }
    }
}
