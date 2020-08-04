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

import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.immutables.value.Value;
import org.jetbrains.annotations.Nullable;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.BaseConfig;
import org.neo4j.graphalgo.config.ConfigurableSeedConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.config.MutatePropertyConfig;
import org.neo4j.graphalgo.config.MutateRelationshipConfig;
import org.neo4j.graphalgo.config.NodeWeightConfig;
import org.neo4j.graphalgo.config.RandomGraphGeneratorConfig;
import org.neo4j.graphalgo.config.RelationshipWeightConfig;
import org.neo4j.graphalgo.config.SeedConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.core.loading.GraphStoreWithConfig;
import org.neo4j.graphalgo.core.loading.ImmutableGraphStoreWithConfig;
import org.neo4j.graphalgo.core.utils.ProgressTimer;
import org.neo4j.graphalgo.core.utils.TerminationFlag;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.MemoryTree;
import org.neo4j.graphalgo.core.utils.mem.MemoryTreeWithDimensions;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.results.MemoryEstimateResult;
import org.neo4j.graphalgo.utils.StringJoining;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;
import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;
import static org.neo4j.graphalgo.config.BaseConfig.SUDO_KEY;
import static org.neo4j.graphalgo.config.ConcurrencyConfig.CONCURRENCY_KEY;
import static org.neo4j.graphalgo.config.ConcurrencyConfig.DEFAULT_CONCURRENCY;
import static org.neo4j.graphalgo.config.GraphCreateConfig.READ_CONCURRENCY_KEY;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public abstract class AlgoBaseProc<
    ALGO extends Algorithm<ALGO, ALGO_RESULT>,
    ALGO_RESULT,
    CONFIG extends AlgoBaseConfig> extends BaseProc {

    protected static final String STATS_DESCRIPTION = "Executes the algorithm and returns result statistics without writing the result to Neo4j.";

    public String algoName() {
        return this.getClass().getSimpleName();
    }

    protected abstract CONFIG newConfig(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper config
    );

    public final CONFIG newConfig(Optional<String> graphName, CypherMapWrapper config) {
        Optional<GraphCreateConfig> maybeImplicitCreate = Optional.empty();
        Collection<String> allowedKeys = new HashSet<>();
        // implicit loading
        if (graphName.isEmpty()) {
            // inherit concurrency from AlgoBaseConfig
            config = config.withNumber(READ_CONCURRENCY_KEY, config.getInt(CONCURRENCY_KEY, DEFAULT_CONCURRENCY));
            GraphCreateConfig createConfig = GraphCreateConfig.createImplicit(username(), config);
            maybeImplicitCreate = Optional.of(createConfig);
            allowedKeys.addAll(createConfig.configKeys());
            CypherMapWrapper configWithoutCreateKeys = config.withoutAny(allowedKeys);
            // check if we have an explicit configured sudo key, as this one is
            // shared between create and algo configs
            Boolean sudoValue = config.getChecked(SUDO_KEY, null, Boolean.class);
            if (sudoValue != null) {
                configWithoutCreateKeys = configWithoutCreateKeys.withBoolean(SUDO_KEY, sudoValue);
            }
            config = configWithoutCreateKeys;
        }
        CONFIG algoConfig = newConfig(username(), graphName, maybeImplicitCreate, config);
        allowedKeys.addAll(algoConfig.configKeys());
        validateConfig(config, allowedKeys);
        return algoConfig;
    }

    // TODO make AlgorithmFactory have a constructor that accepts CONFIG
    protected final ALGO newAlgorithm(
        final Graph graph,
        final CONFIG config,
        final AllocationTracker tracker
    ) {
        TerminationFlag terminationFlag = TerminationFlag.wrap(transaction);
        return algorithmFactory()
            .build(graph, config, tracker, log)
            .withTerminationFlag(terminationFlag);
    }

    protected abstract AlgorithmFactory<ALGO, CONFIG> algorithmFactory();

    protected MemoryTreeWithDimensions memoryEstimation(CONFIG config) {
        MemoryEstimations.Builder estimationBuilder = MemoryEstimations.builder("Memory Estimation");
        GraphDimensions estimateDimensions;

        if (config.implicitCreateConfig().isPresent()) {
            GraphCreateConfig createConfig = config.implicitCreateConfig().get();
            GraphLoader loader = newLoader(createConfig, AllocationTracker.EMPTY);
            GraphStoreFactory<?, ?> graphStoreFactory = loader.graphStoreFactory();
            estimateDimensions = graphStoreFactory.estimationDimensions();

            if (createConfig.nodeCount() >= 0 || createConfig.relationshipCount() >= 0) {
                estimateDimensions = ImmutableGraphDimensions.builder()
                    .from(estimateDimensions)
                    .nodeCount(createConfig.nodeCount())
                    .highestNeoId(createConfig.nodeCount())
                    .maxRelCount(createConfig.relationshipCount())
                    .build();
            }

            estimationBuilder.add("graph", graphStoreFactory.memoryEstimation());
        } else {
            String graphName = config.graphName().orElseThrow(IllegalStateException::new);

            GraphStoreWithConfig graphStoreWithConfig = GraphStoreCatalog.get(
                username(),
                databaseId(),
                graphName
            );
            GraphCreateConfig graphCreateConfig = graphStoreWithConfig.config();
            GraphStore graphStore = graphStoreWithConfig.graphStore();

            // TODO get the dimensions from the graph itself.
            if (graphCreateConfig instanceof RandomGraphGeneratorConfig) {
                estimateDimensions = ImmutableGraphDimensions.builder()
                    .nodeCount(graphCreateConfig.nodeCount())
                    .maxRelCount(((RandomGraphGeneratorConfig) graphCreateConfig).averageDegree() * graphCreateConfig.nodeCount())
                    .build();
            } else {
                Graph filteredGraph = graphStore.getGraph(
                    config.nodeLabelIdentifiers(graphStore),
                    config.internalRelationshipTypes(graphStore),
                    Optional.empty()
                );
                long relCount = filteredGraph.relationshipCount();

                estimateDimensions = ImmutableGraphDimensions.builder()
                    .nodeCount(filteredGraph.nodeCount())
                    .relationshipCounts(Map.of(RelationshipType.ALL_RELATIONSHIPS, relCount))
                    .maxRelCount(relCount)
                    .build();
            }
        }

        estimationBuilder.add("algorithm", algorithmFactory().memoryEstimation(config));

        MemoryTree memoryTree = estimationBuilder.build().estimate(estimateDimensions, config.concurrency());
        return new MemoryTreeWithDimensions(memoryTree, estimateDimensions);
    }

    protected Pair<CONFIG, Optional<String>> processInput(Object graphNameOrConfig, Map<String, Object> configuration) {
        CONFIG config;
        Optional<String> graphName = Optional.empty();

        if (graphNameOrConfig instanceof String) {
            graphName = Optional.of((String) graphNameOrConfig);
            CypherMapWrapper algoConfig = CypherMapWrapper.create(configuration);
            config = newConfig(graphName, algoConfig);

            //TODO: assert that algoConfig is empty or fail
        } else if (graphNameOrConfig instanceof Map) {
            if (!configuration.isEmpty()) {
                throw new IllegalArgumentException(
                    "The second parameter can only used when a graph name is given as first parameter");
            }

            Map<String, Object> implicitConfig = (Map<String, Object>) graphNameOrConfig;
            CypherMapWrapper implicitAndAlgoConfig = CypherMapWrapper.create(implicitConfig);

            config = newConfig(Optional.empty(), implicitAndAlgoConfig);

            //TODO: assert that implicitAndAlgoConfig is empty or fail
        } else {
            throw new IllegalArgumentException(
                "The first parameter must be a graph name or a configuration map, but was: " + graphNameOrConfig
            );
        }

        return Tuples.pair(config, graphName);
    }

    protected Graph createGraph(Pair<CONFIG, Optional<String>> configAndName) {
        return createGraph(getOrCreateGraphStore(configAndName), configAndName.getOne());
    }

    private Graph createGraph(GraphStore graphStore, CONFIG config) {
        Optional<String> weightProperty = config instanceof RelationshipWeightConfig
            ? Optional.ofNullable(((RelationshipWeightConfig) config).relationshipWeightProperty())
            : Optional.empty();

        Collection<NodeLabel> nodeLabels = config.nodeLabelIdentifiers(graphStore);
        Collection<RelationshipType> relationshipTypes = config.internalRelationshipTypes(graphStore);

        return graphStore.getGraph(nodeLabels, relationshipTypes, weightProperty);
    }

    private GraphStore getOrCreateGraphStore(Pair<CONFIG, Optional<String>> configAndName) {
        CONFIG config = configAndName.getOne();
        Optional<String> maybeGraphName = configAndName.getTwo();

        GraphStoreWithConfig graphCandidate;

        if (maybeGraphName.isPresent()) {
            graphCandidate = GraphStoreCatalog.get(username(), databaseId(), maybeGraphName.get());
        } else if (config.implicitCreateConfig().isPresent()) {
            GraphCreateConfig createConfig = config.implicitCreateConfig().get();
            GraphLoader loader = newLoader(createConfig, AllocationTracker.EMPTY);
            GraphStore graphStore = loader.graphStore();

            graphCandidate = ImmutableGraphStoreWithConfig.of(graphStore, createConfig);
        } else {
            throw new IllegalStateException("There must be either a graph name or an implicit create config");
        }

        validate(graphCandidate, config);
        return graphCandidate.graphStore();
    }

    private void validate(GraphStoreWithConfig graphStoreWithConfig, CONFIG config) {
        GraphStore graphStore = graphStoreWithConfig.graphStore();
        GraphCreateConfig graphCreateConfig = graphStoreWithConfig.config();

        if (graphCreateConfig instanceof GraphCreateFromCypherConfig) {
            return;
        }

        Collection<NodeLabel> filterLabels = config.nodeLabelIdentifiers(graphStore);
        if (config instanceof SeedConfig) {
            String seedProperty = ((SeedConfig) config).seedProperty();
            if (seedProperty != null && !graphStore.hasNodeProperty(filterLabels, seedProperty)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Seed property `%s` not found in graph with node properties: %s",
                    seedProperty,
                    graphStore.nodePropertyKeys().values()
                ));
            }
        }
        if (config instanceof ConfigurableSeedConfig) {
            ConfigurableSeedConfig configurableSeedConfig = (ConfigurableSeedConfig) config;
            String seedProperty = configurableSeedConfig.seedProperty();
            if (seedProperty != null && !graphStore.hasNodeProperty(filterLabels, seedProperty)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "`%s`: `%s` not found in graph with node properties: %s",
                    configurableSeedConfig.propertyNameOverride(),
                    seedProperty,
                    graphStore.nodePropertyKeys().values()
                ));
            }
        }
        if (config instanceof NodeWeightConfig) {
            String weightProperty = ((NodeWeightConfig) config).nodeWeightProperty();
            if (weightProperty != null && !graphStore.hasNodeProperty(filterLabels, weightProperty)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Node weight property `%s` not found in graph with node properties: %s in all node labels: %s",
                    weightProperty,
                    graphStore.nodePropertyKeys(filterLabels),
                    StringJoining.join(filterLabels.stream().map(NodeLabel::name))
                ));
            }
        }
        if (config instanceof RelationshipWeightConfig) {

            String weightProperty = ((RelationshipWeightConfig) config).relationshipWeightProperty();
            Collection<RelationshipType> internalRelationshipTypes = config.internalRelationshipTypes(graphStore);
            if (weightProperty != null && !graphStore.hasRelationshipProperty(internalRelationshipTypes, weightProperty)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Relationship weight property `%s` not found in graph with relationship properties: %s in all relationship types: %s",
                    weightProperty,
                    graphStore.relationshipPropertyKeys(internalRelationshipTypes),
                    StringJoining.join(internalRelationshipTypes.stream().map(RelationshipType::name))
                ));
            }
        }

        if (config instanceof MutatePropertyConfig) {
            MutatePropertyConfig mutateConfig = (MutatePropertyConfig) config;
            String mutateProperty = mutateConfig.mutateProperty();

            if (mutateProperty != null && graphStore.hasNodeProperty(filterLabels, mutateProperty)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Node property `%s` already exists in the in-memory graph.",
                    mutateProperty
                ));
            }
        }

        if (config instanceof MutateRelationshipConfig) {
            String mutateRelationshipType = ((MutateRelationshipConfig) config).mutateRelationshipType();
            if (mutateRelationshipType != null && graphStore.hasRelationshipType(RelationshipType.of(mutateRelationshipType))) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Relationship type `%s` already exists in the in-memory graph.",
                    mutateRelationshipType
                ));
            }
        }

        validateConfigs(graphCreateConfig, config);
    }

    protected void validateConfigs(GraphCreateConfig graphCreateConfig, CONFIG config) { }

    protected void validateIsUndirectedGraph(GraphCreateConfig graphCreateConfig, CONFIG config) {
        graphCreateConfig.accept(new GraphCreateConfig.Visitor() {
            @Override
            public void visit(GraphCreateFromStoreConfig storeConfig) {
                storeConfig.relationshipProjections().projections().entrySet().stream()
                    .filter(entry -> config.relationshipTypes().equals(Collections.singletonList(PROJECT_ALL)) ||
                                     config.relationshipTypes().contains(entry.getKey().name()))
                    .filter(entry -> entry.getValue().orientation() != Orientation.UNDIRECTED)
                    .forEach(entry -> {
                        throw new IllegalArgumentException(formatWithLocale(
                            "Procedure requires relationship projections to be UNDIRECTED. Projection for `%s` uses orientation `%s`",
                            entry.getKey().name,
                            entry.getValue().orientation()
                        ));
                    });

            }
        });
    }

    /**
     * Validates that {@link Orientation#UNDIRECTED} is not mixed with {@link Orientation#NATURAL}
     * and {@link Orientation#REVERSE}. If a relationship type filter is present in the algorithm
     * config, only those relationship projections are considered in the validation.
     */
    protected void validateOrientationCombinations(GraphCreateConfig graphCreateConfig, CONFIG algorithmConfig) {
        graphCreateConfig.accept(new GraphCreateConfig.Visitor() {
            @Override
            public void visit(GraphCreateFromStoreConfig storeConfig) {
                var filteredProjections = storeConfig
                    .relationshipProjections()
                    .projections()
                    .entrySet()
                    .stream()
                    .filter(entry -> algorithmConfig.relationshipTypes().equals(Collections.singletonList(PROJECT_ALL)) ||
                                     algorithmConfig.relationshipTypes().contains(entry.getKey().name()))
                    .collect(toList());

                boolean allUndirected = filteredProjections
                    .stream()
                    .allMatch(entry -> entry.getValue().orientation() == Orientation.UNDIRECTED);

                boolean anyUndirected = filteredProjections
                    .stream()
                    .anyMatch(entry -> entry.getValue().orientation() == Orientation.UNDIRECTED);

                if (anyUndirected && !allUndirected) {
                    throw new IllegalArgumentException(formatWithLocale(
                        "Combining UNDIRECTED orientation with NATURAL or REVERSE is not supported. Found projections: %s.",
                        StringJoining.join(filteredProjections
                            .stream()
                            .map(entry -> formatWithLocale("%s (%s)", entry.getKey().name, entry.getValue().orientation()))
                            .sorted())
                    ));
                }
            }
        });
    }

    protected ComputationResult<ALGO, ALGO_RESULT, CONFIG> compute(
        Object graphNameOrConfig,
        Map<String, Object> configuration
    ) {
        return compute(graphNameOrConfig, configuration, true, true);
    }

    protected ComputationResult<ALGO, ALGO_RESULT, CONFIG> compute(
        Object graphNameOrConfig,
        Map<String, Object> configuration,
        boolean releaseAlgorithm,
        boolean releaseTopology
    ) {
        ImmutableComputationResult.Builder<ALGO, ALGO_RESULT, CONFIG> builder = ImmutableComputationResult.builder();
        AllocationTracker tracker = AllocationTracker.create();

        Pair<CONFIG, Optional<String>> input = processInput(graphNameOrConfig, configuration);
        CONFIG config = input.getOne();

        validateMemoryUsageIfImplemented(config);

        GraphStore graphStore;
        Graph graph;

        try (ProgressTimer timer = ProgressTimer.start(builder::createMillis)) {
            graphStore = getOrCreateGraphStore(input);
            graph = createGraph(graphStore, config);
        }

        if (graph.isEmpty()) {
            return builder
                .isGraphEmpty(true)
                .graph(graph)
                .graphStore(graphStore)
                .config(config)
                .tracker(tracker)
                .computeMillis(0)
                .result(null)
                .algorithm(null)
                .build();
        }

        ALGO algo = newAlgorithm(graph, config, tracker);

        ALGO_RESULT result = runWithExceptionLogging(
            "Computation failed",
            () -> {
                try (ProgressTimer ignored = ProgressTimer.start(builder::computeMillis)) {
                    return algo.compute();
                }
            }
        );

        log.info(algoName() + ": overall memory usage %s", tracker.getUsageString());

        if (releaseAlgorithm) {
            algo.release();
        }
        if (releaseTopology) {
            graph.releaseTopology();
        }

        return builder
            .graph(graph)
            .graphStore(graphStore)
            .tracker(AllocationTracker.EMPTY)
            .algorithm(algo)
            .result(result)
            .config(config)
            .build();
    }

    protected NodeProperties getNodeProperties(
        ComputationResult<ALGO, ALGO_RESULT, CONFIG> computationResult
    ) {
        throw new UnsupportedOperationException(
            "Procedure needs to implement org.neo4j.graphalgo.BaseAlgoProc.nodePropertyTranslator");
    }

    private void validateMemoryUsageIfImplemented(CONFIG config) {
        var sudoImplicitCreate = config.implicitCreateConfig().map(BaseConfig::sudo).orElse(false);

        if (sudoImplicitCreate) {
            log.debug("Sudo mode: Won't check for available memory.");
            return;
        }

        tryValidateMemoryUsage(config, this::memoryEstimation);
    }

    protected Stream<MemoryEstimateResult> computeEstimate(
        Object graphNameOrConfig,
        Map<String, Object> configuration
    ) {
        Pair<CONFIG, Optional<String>> configAndGraphName = processInput(
            graphNameOrConfig,
            configuration
        );

        MemoryTreeWithDimensions memoryTreeWithDimensions = memoryEstimation(configAndGraphName.getOne());
        return Stream.of(
            new MemoryEstimateResult(memoryTreeWithDimensions)
        );
    }

    @ValueClass
    public interface ComputationResult<A extends Algorithm<A, RESULT>, RESULT, CONFIG extends AlgoBaseConfig> {
        long createMillis();

        long computeMillis();

        @Nullable
        A algorithm();

        @Nullable
        RESULT result();

        Graph graph();

        GraphStore graphStore();

        AllocationTracker tracker();

        CONFIG config();

        @Value.Default
        default boolean isGraphEmpty() {
            return false;
        }
    }
}
