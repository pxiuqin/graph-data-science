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
import org.neo4j.graphalgo.StatsProc;
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

public class BetweennessCentralityStatsProc extends StatsProc<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityStatsProc.StatsResult, BetweennessCentralityStatsConfig> {

    @Procedure(value = "gds.betweenness.stats", mode = READ)
    @Description(BETWEENNESS_DESCRIPTION)
    public Stream<StatsResult> stats(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return stats(compute(graphNameOrConfig, configuration));
    }

    @Procedure(value = "gds.betweenness.stats.estimate", mode = READ)
    @Description(BETWEENNESS_DESCRIPTION)
    public Stream<MemoryEstimateResult> estimate(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return computeEstimate(graphNameOrConfig, configuration);
    }

    @Override
    protected BetweennessCentralityStatsConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return BetweennessCentralityStatsConfig.of(username, graphName, maybeImplicitCreate, config);
    }

    @Override
    protected void validateConfigs(GraphCreateConfig graphCreateConfig, BetweennessCentralityStatsConfig config) {
        validateOrientationCombinations(graphCreateConfig, config);
    }

    @Override
    protected AlgorithmFactory<BetweennessCentrality, BetweennessCentralityStatsConfig> algorithmFactory() {
        return BetweennessCentralityProc.algorithmFactory();
    }

    @Override
    protected NodeProperties getNodeProperties(ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityStatsConfig> computationResult) {
        return BetweennessCentralityProc.nodeProperties(computationResult);
    }

    @Override
    protected AbstractResultBuilder<StatsResult> resultBuilder(ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, BetweennessCentralityStatsConfig> computeResult) {
        return BetweennessCentralityProc.resultBuilder(new StatsResult.Builder(), computeResult, callContext);
    }

    public static class StatsResult {

        public final double minimumScore;
        public final double maximumScore;
        public final double scoreSum;

        public final long postProcessingMillis;
        public final long createMillis;
        public final long computeMillis;

        public final Map<String, Object> configuration;

        StatsResult(
            long createMillis,
            long computeMillis,
            long postProcessingMillis,
            double minimumScore,
            double maximumScore,
            double scoreSum,
            Map<String, Object> configuration
        ) {
            this.createMillis = createMillis;
            this.computeMillis = computeMillis;
            this.postProcessingMillis = postProcessingMillis;

            this.minimumScore = minimumScore;
            this.maximumScore = maximumScore;
            this.scoreSum = scoreSum;
            this.configuration = configuration;
        }

        static final class Builder extends BetweennessCentralityProc.BetweennessCentralityResultBuilder<StatsResult> {

            @Override
            public StatsResult build() {
                return new StatsResult(
                    createMillis,
                    computeMillis,
                    postProcessingMillis,
                    minimumScore,
                    maximumScore,
                    scoreSum,
                    config.toMap()
                );
            }
        }
    }
}
