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
package org.neo4j.gds.embeddings.graphsage.algo;

import org.immutables.value.Value;
import org.neo4j.gds.embeddings.graphsage.ActivationFunction;
import org.neo4j.gds.embeddings.graphsage.Aggregator;
import org.neo4j.gds.embeddings.graphsage.LayerConfig;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.BatchSizeConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.IterationsConfig;
import org.neo4j.graphalgo.config.NodePropertiesConfig;
import org.neo4j.graphalgo.config.ToleranceConfig;
import org.neo4j.graphalgo.config.TrainConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

@ValueClass
@Configuration("GraphSageTrainConfigImpl")
@SuppressWarnings("immutables:subtype")
public interface GraphSageTrainConfig extends AlgoBaseConfig, TrainConfig, BatchSizeConfig, IterationsConfig, ToleranceConfig, NodePropertiesConfig {

    @Value.Default
    default int embeddingSize() {
        return 64;
    }

    @Value.Default
    default List<Long> sampleSizes() {
        return List.of(25L, 10L);
    }

    @Configuration.ConvertWith("org.neo4j.gds.embeddings.graphsage.Aggregator.AggregatorType#parse")
    @Configuration.ToMapValue("org.neo4j.gds.embeddings.graphsage.Aggregator.AggregatorType#toString")
    @Value.Default
    default Aggregator.AggregatorType aggregator() {
        return Aggregator.AggregatorType.MEAN;
    }

    @Configuration.ConvertWith("org.neo4j.gds.embeddings.graphsage.ActivationFunction#parse")
    @Configuration.ToMapValue("org.neo4j.gds.embeddings.graphsage.ActivationFunction#toString")
    @Value.Default
    default ActivationFunction activationFunction() {
        return ActivationFunction.SIGMOID;
    }

    @Value.Default
    @Override
    default double tolerance() {
        return 1e-4;
    }

    @Value.Default
    default double learningRate() {
        return 0.1;
    }

    @Value.Default
    default int epochs() {
        return 1;
    }

    @Value.Default
    @Override
    default int maxIterations() {
        return 10;
    }

    @Value.Default
    default int searchDepth() {
        return 5;
    }

    @Value.Default
    default int negativeSampleWeight() {
        return 20;
    }

    @Value.Default
    default boolean degreeAsProperty() {
        return false;
    }

    @Configuration.Ignore
    default Collection<LayerConfig> layerConfigs() {
        Collection<LayerConfig> result = new ArrayList<>(sampleSizes().size());
        for (int i = 0; i < sampleSizes().size(); i++) {
            LayerConfig layerConfig = LayerConfig.builder()
                .aggregatorType(aggregator())
                .activationFunction(activationFunction())
                .rows(embeddingSize())
                .cols(i == 0 ? featuresSize() : embeddingSize())
                .sampleSize(sampleSizes().get(i))
                .build();

            result.add(layerConfig);
        }

        return result;
    }

    @Configuration.Ignore
    default int featuresSize() {
        return nodePropertyNames().size() + (degreeAsProperty() ? 1 : 0);
    }

    @Value.Check
    default void validate() {
        if (nodePropertyNames().isEmpty() && !degreeAsProperty()) {
            throw new IllegalArgumentException(
                "GraphSage requires at least one property. Either `nodePropertyNames` or `degreeAsProperty` must be set."
            );
        }
    }

    static GraphSageTrainConfig of(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper userInput
    ) {
        return new GraphSageTrainConfigImpl(
            graphName,
            maybeImplicitCreate,
            username,
            userInput
        );
    }

    static GraphSageTrainConfig of(
        String modelName,
        ActivationFunction activationFunction,
        Aggregator.AggregatorType aggregator,
        int batchSize,
        int embeddingSize,
        List<String> nodePropertyNames,
        double tolerance
    ) {
        return ImmutableGraphSageTrainConfig.builder()
            .modelName(modelName)
            .activationFunction(activationFunction)
            .aggregator(aggregator)
            .batchSize(batchSize)
            .embeddingSize(embeddingSize)
            .nodePropertyNames(nodePropertyNames)
            .tolerance(tolerance)
            .build();
    }

}
