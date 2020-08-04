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

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.core.utils.ProgressTimer;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicDoubleArray;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;

final class BetweennessCentralityProc {

    static final String BETWEENNESS_DESCRIPTION = "Betweenness centrality measures the relative information flow that passes through a node.";

    private BetweennessCentralityProc() {}

    static <CONFIG extends BetweennessCentralityBaseConfig> NodeProperties nodeProperties(AlgoBaseProc.ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, CONFIG> computeResult) {
        return computeResult.result().asNodeProperties();
    }

    static <CONFIG extends BetweennessCentralityBaseConfig> AlgorithmFactory<BetweennessCentrality, CONFIG> algorithmFactory() {
        return new BetweennessCentralityFactory<>();
    }

    static <PROC_RESULT, CONFIG extends BetweennessCentralityBaseConfig> AbstractResultBuilder<PROC_RESULT> resultBuilder(
        BetweennessCentralityResultBuilder<PROC_RESULT> procResultBuilder,
        AlgoBaseProc.ComputationResult<BetweennessCentrality, HugeAtomicDoubleArray, CONFIG> computeResult,
        ProcedureCallContext callContext
    ) {
        var computeStatistics = callContext.outputFields().anyMatch(f ->
            f.equalsIgnoreCase("minimumScore") ||
            f.equalsIgnoreCase("maximumScore") ||
            f.equalsIgnoreCase("scoreSum")
        );

        var result = computeResult.result();
        if (result != null && computeStatistics) {
            ProgressTimer timer = ProgressTimer.start();
            procResultBuilder = computeStatistics(procResultBuilder, result);
            procResultBuilder.withPostProcessingMillis(timer.getDuration());
        }

        return procResultBuilder.withConfig(computeResult.config());
    }

    private static <PROC_RESULT> BetweennessCentralityResultBuilder<PROC_RESULT> computeStatistics(
        BetweennessCentralityResultBuilder<PROC_RESULT> procResultBuilder,
        HugeAtomicDoubleArray result
    ) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        for (long i = 0; i < result.size(); i++) {
            double c = result.get(i);
            if (c < min) {
                min = c;
            }
            if (c > max) {
                max = c;
            }
            sum += c;
        }
        return procResultBuilder
            .minimumScore(min)
            .maximumScore(max)
            .scoreSum(sum);
    }

    abstract static class BetweennessCentralityResultBuilder<PROC_RESULT> extends AbstractResultBuilder<PROC_RESULT> {

        double minimumScore, maximumScore, scoreSum = -1;
        long postProcessingMillis = 0;

        BetweennessCentralityResultBuilder<PROC_RESULT> minimumScore(double minimumScore) {
            this.minimumScore = minimumScore;
            return this;
        }

        BetweennessCentralityResultBuilder<PROC_RESULT> maximumScore(double maximumScore) {
            this.maximumScore = maximumScore;
            return this;
        }

        BetweennessCentralityResultBuilder<PROC_RESULT> scoreSum(double scoreSum) {
            this.scoreSum = scoreSum;
            return this;
        }

        BetweennessCentralityResultBuilder<PROC_RESULT> withPostProcessingMillis(long postProcessingMillis) {
            this.postProcessingMillis = postProcessingMillis;
            return this;
        }
    }
}
