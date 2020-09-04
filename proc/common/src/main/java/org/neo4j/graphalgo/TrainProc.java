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
package org.neo4j.graphalgo;

import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.config.TrainConfig;
import org.neo4j.graphalgo.core.model.Model;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.neo4j.graphalgo.config.TrainConfig.MODEL_NAME_KEY;
import static org.neo4j.graphalgo.config.TrainConfig.MODEL_TYPE_KEY;

public abstract class TrainProc<ALGO extends Algorithm<ALGO, Model<TRAIN_RESULT, TRAIN_CONFIG>>,
    TRAIN_RESULT,
    TRAIN_CONFIG extends TrainConfig & AlgoBaseConfig>
    extends AlgoBaseProc<ALGO, Model<TRAIN_RESULT, TRAIN_CONFIG>, TRAIN_CONFIG> {

    protected TrainResult trainResult(ComputationResult<ALGO, Model<TRAIN_RESULT, TRAIN_CONFIG>, TRAIN_CONFIG> computationResult) {
        return new TrainResult(
            computationResult.result(),
            computationResult.computeMillis(),
            computationResult.graph().nodeCount(),
            computationResult.graph().relationshipCount()
        );
    }

    public static class TrainResult {

        public final String graphName;
        public Map<String, Object> graphCreateConfig;
        public final Map<String, Object> modelInfo;
        public final Map<String, Object> configuration;
        public final long trainMillis;

        public <TRAIN_RESULT, TRAIN_CONFIG extends TrainConfig & AlgoBaseConfig> TrainResult(
            Model<TRAIN_RESULT, TRAIN_CONFIG> trainedModel,
            long trainMillis,
            long nodeCount,
            long relationshipCount
        ) {

            TRAIN_CONFIG trainConfig = trainedModel.trainConfig();

            this.graphName = trainConfig.graphName().orElse(null);
            this.graphCreateConfig = anonymousGraphResult(
                nodeCount,
                relationshipCount,
                trainConfig
            );
            this.modelInfo = new HashMap<>();
            modelInfo.put(MODEL_NAME_KEY, trainedModel.name());
            modelInfo.put(MODEL_TYPE_KEY, trainedModel.algoType());
            this.configuration = trainConfig.toMap();
            this.trainMillis = trainMillis;
        }

        private <TRAIN_CONFIG extends TrainConfig & AlgoBaseConfig> Map<String, Object> anonymousGraphResult(
            long nodeCount,
            long relationshipCount,
            TRAIN_CONFIG trainConfig
        ) {
            final AtomicReference<GraphCreateResult> anonymousGraphBuilder = new AtomicReference<>();
            trainConfig.implicitCreateConfig().ifPresent(graphCreateConfig -> graphCreateConfig.accept(
                new GraphCreateConfig.Visitor() {
                    @Override
                    public void visit(GraphCreateFromStoreConfig storeConfig) {
                        var result = GraphCreateNativeResult.builder()
                            .nodeCount(nodeCount)
                            .relationshipCount(relationshipCount)
                            .nodeProjection(storeConfig.nodeProjections().toObject())
                            .relationshipProjection(storeConfig.relationshipProjections().toObject())
                            .build();
                        anonymousGraphBuilder.set(result);
                    }

                    @Override
                    public void visit(GraphCreateFromCypherConfig cypherConfig) {
                        var result = GraphCreateCypherResult.builder()
                            .nodeCount(nodeCount)
                            .relationshipCount(relationshipCount)
                            .nodeQuery(cypherConfig.nodeQuery())
                            .relationshipQuery(cypherConfig.relationshipQuery())
                            .build();
                        anonymousGraphBuilder.set(result);
                    }
                }
            ));
            return Optional
                .ofNullable(anonymousGraphBuilder.get())
                .map(GraphCreateResult::toMap)
                .orElse(Collections.emptyMap());
        }
    }

    @ValueClass
    interface GraphCreateResult {
        long nodeCount();
        long relationshipCount();

        default Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("nodeCount", nodeCount());
            map.put("relationshipCount", relationshipCount());
            return map;
        }
    }


    @ValueClass
    interface GraphCreateNativeResult extends GraphCreateResult {
        Map<String, Object> nodeProjection();
        Map<String, Object> relationshipProjection();

        static ImmutableGraphCreateNativeResult.Builder builder() {
            return ImmutableGraphCreateNativeResult.builder();
        }

        default Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("nodeProjection", nodeProjection());
            map.put("relationshipProjection", relationshipProjection());
            map.put("nodeCount", nodeCount());
            map.put("relationshipCount", relationshipCount());
            return map;
        }
    }

    @ValueClass
    interface GraphCreateCypherResult extends GraphCreateResult {
        String nodeQuery();
        String relationshipQuery();

        static ImmutableGraphCreateCypherResult.Builder builder() {
            return ImmutableGraphCreateCypherResult.builder();
        }

        default Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("nodeQuery", nodeQuery());
            map.put("relationshipQuery", relationshipQuery());
            map.put("nodeCount", nodeCount());
            map.put("relationshipCount", relationshipCount());
            return map;
        }
    }

}
