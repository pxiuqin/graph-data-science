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
package org.neo4j.graphalgo.centrality;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.AlphaAlgorithmFactory;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.centrality.degreecentrality.DegreeCentrality;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.graphalgo.results.CentralityScore;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.neo4j.procedure.Mode.READ;
import static org.neo4j.procedure.Mode.WRITE;

public class DegreeCentralityProc extends AlgoBaseProc<DegreeCentrality, DegreeCentrality, DegreeCentralityConfig> {

    private static final String DESCRIPTION = "Degree centrality measures the number of incoming and outgoing relationships from a node.";

    @Procedure(value = "gds.alpha.degree.write", mode = WRITE)
    @Description(DESCRIPTION)
    public Stream<CentralityScore.Stats> write(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        ComputationResult<DegreeCentrality, DegreeCentrality, DegreeCentralityConfig> computeResult = compute(
            graphNameOrConfig,
            configuration
        );

        return write(computeResult);
    }

    @Procedure(name = "gds.alpha.degree.stream", mode = READ)
    @Description(DESCRIPTION)
    public Stream<CentralityScore> stream(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        ComputationResult<DegreeCentrality, DegreeCentrality, DegreeCentralityConfig> computeResult = compute(
            graphNameOrConfig,
            configuration
        );
        return CentralityUtils.streamResults(computeResult.graph(), computeResult.algorithm().result());
    }

    private Stream<CentralityScore.Stats> write(
        ComputationResult<DegreeCentrality, DegreeCentrality, DegreeCentralityConfig> computeResult
    ) {
        Graph graph = computeResult.graph();
        if (graph.isEmpty()) {
            graph.release();
            return Stream.of(new CentralityScore.Stats(
                    0,
                    0,
                    computeResult.createMillis(),
                    0,
                    computeResult.config().writeProperty()
                )
            );
        }

        DegreeCentralityConfig config = computeResult.config();
        DegreeCentrality algorithm = computeResult.algorithm();

        AbstractResultBuilder<CentralityScore.Stats> builder = new CentralityScore.Stats.Builder()
            .withNodeCount(graph.nodeCount());


        CentralityUtils.write(
            api,
            log,
            computeResult.graph(),
            algorithm.getTerminationFlag(),
            algorithm.result(),
            config,
            builder
        );

        builder.withCreateMillis(computeResult.createMillis())
                .withComputeMillis(computeResult.computeMillis());

        graph.release();
        return Stream.of(builder.build());
    }

    @Override
    protected DegreeCentralityConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper userInput
    ) {
        return new DegreeCentralityConfigImpl(graphName, maybeImplicitCreate, username, userInput);
    }

    @Override
    protected AlgorithmFactory<DegreeCentrality, DegreeCentralityConfig> algorithmFactory() {
        return (AlphaAlgorithmFactory<DegreeCentrality, DegreeCentralityConfig>) (graph, configuration, tracker, log) ->
            new DegreeCentrality(
                graph,
                Pools.DEFAULT,
                configuration.concurrency(),
                configuration.isWeighted(),
                tracker
            );
    }
}
