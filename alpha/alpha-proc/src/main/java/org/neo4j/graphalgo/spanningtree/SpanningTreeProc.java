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
package org.neo4j.graphalgo.spanningtree;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.AlphaAlgorithmFactory;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.utils.ProgressTimer;
import org.neo4j.graphalgo.core.write.RelationshipExporter;
import org.neo4j.graphalgo.impl.spanningTrees.Prim;
import org.neo4j.graphalgo.impl.spanningTrees.SpanningGraph;
import org.neo4j.graphalgo.impl.spanningTrees.SpanningTree;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.utils.InputNodeValidator.validateStartNode;
import static org.neo4j.procedure.Mode.WRITE;

// TODO: Always undirected
public class SpanningTreeProc extends AlgoBaseProc<Prim, SpanningTree, SpanningTreeConfig> {

    private static final String MIN_DESCRIPTION =
        "Minimum weight spanning tree visits all nodes that are in the same connected component as the starting node, " +
        "and returns a spanning tree of all nodes in the component where the total weight of the relationships is minimized.";

    private static final String MAX_DESCRIPTION =
        "Maximum weight spanning tree visits all nodes that are in the same connected component as the starting node, " +
        "and returns a spanning tree of all nodes in the component where the total weight of the relationships is maximized.";

    static DoubleUnaryOperator minMax;

    @Procedure(value = "gds.alpha.spanningTree.write", mode = WRITE)
    @Description(MIN_DESCRIPTION)
    public Stream<Prim.Result> spanningTree(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        minMax = Prim.MIN_OPERATOR;
        return computeAndWrite(graphNameOrConfig, configuration);
    }

    @Procedure(value = "gds.alpha.spanningTree.minimum.write", mode = WRITE)
    @Description(MIN_DESCRIPTION)
    public Stream<Prim.Result> minimumSpanningTree(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        minMax = Prim.MIN_OPERATOR;
        return computeAndWrite(graphNameOrConfig, configuration);
    }

    @Procedure(value = "gds.alpha.spanningTree.maximum.write", mode = WRITE)
    @Description(MAX_DESCRIPTION)
    public Stream<Prim.Result> maximumSpanningTree(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        minMax = Prim.MAX_OPERATOR;
        return computeAndWrite(graphNameOrConfig, configuration);
    }

    private Stream<Prim.Result> computeAndWrite(Object graphNameOrConfig, Map<String, Object> configuration) {
        ComputationResult<Prim, SpanningTree, SpanningTreeConfig> computationResult = compute(graphNameOrConfig, configuration);

        Graph graph = computationResult.graph();
        Prim prim = computationResult.algorithm();
        SpanningTree spanningTree = computationResult.result();
        SpanningTreeConfig config = computationResult.config();

        Prim.Builder builder = new Prim.Builder();

        if (graph.isEmpty()) {
            graph.release();
            return Stream.of(builder.build());
        }

        builder.withEffectiveNodeCount(spanningTree.effectiveNodeCount);
        try (ProgressTimer ignored = ProgressTimer.start(builder::withWriteMillis)) {
            RelationshipExporter.of(
                api,
                new SpanningGraph(graph, spanningTree),
                prim.getTerminationFlag()
            )
                .withLog(log)
                .build()
                .write(config.writeProperty(), config.weightWriteProperty());

        }
        builder.withComputeMillis(computationResult.computeMillis());
        builder.withCreateMillis(computationResult.createMillis());
        return Stream.of(builder.build());
    }

    @Override
    protected SpanningTreeConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return SpanningTreeConfig.of(username, graphName, maybeImplicitCreate, config);
    }

    @Override
    protected AlgorithmFactory<Prim, SpanningTreeConfig> algorithmFactory() {
        return (AlphaAlgorithmFactory<Prim, SpanningTreeConfig>) (graph, configuration, tracker, log) -> {
            validateStartNode(configuration.startNodeId(), graph);
            return new Prim(graph, graph, minMax, configuration.startNodeId());
        };
    }
}
