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
package org.neo4j.graphalgo.shortestpaths;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.AlphaAlgorithmFactory;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.impl.shortestpaths.ShortestPathAStar;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.utils.InputNodeValidator.validateEndNode;
import static org.neo4j.graphalgo.utils.InputNodeValidator.validateStartNode;
import static org.neo4j.procedure.Mode.READ;

public class ShortestPathAStarProc extends AlgoBaseProc<ShortestPathAStar, ShortestPathAStar, ShortestPathAStarConfig> {

    private static final String DESCRIPTION = "The A* algorithm is a search algorithm and improves on the classic Dijkstra algorithm.";

    @Procedure(name = "gds.alpha.shortestPath.astar.stream", mode = READ)
    @Description(DESCRIPTION)
    public Stream<ShortestPathAStar.Result> astarStream(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        ComputationResult<ShortestPathAStar, ShortestPathAStar, ShortestPathAStarConfig> computationResult = compute(
            graphNameOrConfig,
            configuration,
            false,
            false
        );

        Graph graph = computationResult.graph();
        if (graph.isEmpty()) {
            graph.release();
            return Stream.empty();
        }

        ShortestPathAStar algo = computationResult.algorithm();
        return algo.resultStream();
    }

    @Override
    protected ShortestPathAStarConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return ShortestPathAStarConfig.of(graphName, maybeImplicitCreate, username, config);
    }

    @Override
    protected AlgorithmFactory<ShortestPathAStar, ShortestPathAStarConfig> algorithmFactory() {
        return (AlphaAlgorithmFactory<ShortestPathAStar, ShortestPathAStarConfig>) (graph, configuration, tracker, log) -> {
            validateStartNode(configuration.startNodeId(), graph);
            validateEndNode(configuration.endNodeId(), graph);
            return new ShortestPathAStar(
                graph,
                configuration.startNodeId(),
                configuration.endNodeId(),
                graph.nodeProperties(configuration.propertyKeyLat()),
                graph.nodeProperties(configuration.propertyKeyLon())
            );
        };
    }
}
