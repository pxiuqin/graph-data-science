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
package org.neo4j.graphalgo.beta.k1coloring;

import org.neo4j.graphalgo.AlgorithmFactory;
import org.neo4j.graphalgo.StreamProc;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.utils.paged.HugeLongArray;
import org.neo4j.graphalgo.results.MemoryEstimateResult;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.beta.k1coloring.K1ColoringProc.K1_COLORING_DESCRIPTION;
import static org.neo4j.procedure.Mode.READ;

public class K1ColoringStreamProc extends StreamProc<K1Coloring, HugeLongArray, K1ColoringStreamProc.StreamResult, K1ColoringStreamConfig> {

    @Procedure(name = "gds.beta.k1coloring.stream", mode = READ)
    @Description(K1_COLORING_DESCRIPTION)
    public Stream<StreamResult> stream(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return stream(compute(graphNameOrConfig, configuration));
    }

    @Procedure(value = "gds.beta.k1coloring.stream.estimate", mode = READ)
    @Description(ESTIMATE_DESCRIPTION)
    public Stream<MemoryEstimateResult> estimate(
        @Name(value = "graphName") Object graphNameOrConfig,
        @Name(value = "configuration", defaultValue = "{}") Map<String, Object> configuration
    ) {
        return computeEstimate(graphNameOrConfig, configuration);
    }

    @Override
    protected K1ColoringStreamConfig newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    ) {
        return K1ColoringStreamConfig.of(username, graphName, maybeImplicitCreate, config);
    }

    @Override
    protected AlgorithmFactory<K1Coloring, K1ColoringStreamConfig> algorithmFactory() {
        return new K1ColoringFactory<>();
    }

    @Override
    protected StreamResult streamResult(
        long originalNodeId, long internalNodeId, NodeProperties nodeProperties
    ) {
        return new StreamResult(originalNodeId, nodeProperties.longValue(internalNodeId));
    }

    @Override
    protected NodeProperties getNodeProperties(ComputationResult<K1Coloring, HugeLongArray, K1ColoringStreamConfig> computationResult) {
        return K1ColoringProc.nodeProperties(computationResult);
    }

    public static class StreamResult {
        public final long nodeId;
        public final long color;

        StreamResult(long nodeId, long color) {
            this.nodeId = nodeId;
            this.color = color;
        }
    }
}
