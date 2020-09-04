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
package org.neo4j.graphalgo.core.utils.export;

import org.eclipse.collections.impl.tuple.Tuples;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.TestOnly;
import org.neo4j.common.Validator;
import org.neo4j.configuration.Config;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.NodeMapping;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.RelationshipIterator;
import org.neo4j.graphalgo.compat.Neo4jProxy;
import org.neo4j.graphalgo.core.Settings;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeIntArray;
import org.neo4j.internal.batchimport.AdditionalInitialIds;
import org.neo4j.internal.batchimport.BatchImporterFactory;
import org.neo4j.internal.batchimport.Configuration;
import org.neo4j.internal.batchimport.ImportLogic;
import org.neo4j.internal.batchimport.input.Collector;
import org.neo4j.internal.batchimport.input.Input;
import org.neo4j.internal.batchimport.staging.ExecutionMonitors;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.Neo4jLayout;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.kernel.impl.store.format.RecordFormatSelector;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.kernel.lifecycle.LifeSupport;
import org.neo4j.logging.internal.LogService;
import org.neo4j.logging.internal.NullLogService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;
import static org.neo4j.io.ByteUnit.mebiBytes;
import static org.neo4j.kernel.impl.scheduler.JobSchedulerFactory.createScheduler;

public class GraphStoreExport {

    private final GraphStore graphStore;

    private final Path neo4jHome;

    private final GraphStoreExportConfig config;

    private final FileSystemAbstraction fs;

    public GraphStoreExport(
        GraphStore graphStore,
        GraphDatabaseAPI api,
        GraphStoreExportConfig config
    ) {
        this.graphStore = graphStore;
        this.neo4jHome = Neo4jProxy.homeDirectory(api.databaseLayout());
        this.config = config;
        this.fs = api.getDependencyResolver().resolveDependency(FileSystemAbstraction.class);
    }

    public ImportedProperties run() {
        return run(false);
    }

    /**
     * Runs with default configuration geared towards
     * unit/integration test environments, for example,
     * lower default buffer sizes.
     */
    @TestOnly
    public void runFromTests() {
        run(true);
    }

    private ImportedProperties run(boolean defaultSettingsSuitableForTests) {
        DIRECTORY_IS_WRITABLE.validate(neo4jHome);
        var databaseConfig = Config.defaults(Settings.neo4jHome(), neo4jHome);
        var databaseLayout = Neo4jLayout.of(databaseConfig).databaseLayout(config.dbName());
        var importConfig = getImportConfig(defaultSettingsSuitableForTests);

        var lifeSupport = new LifeSupport();

       try {
            LogService logService;
            if (config.enableDebugLog()) {
                var storeInternalLogPath = databaseConfig.get(Settings.storeInternalLogPath());
                logService = Neo4jProxy.logProviderForStoreAndRegister(storeInternalLogPath, fs, lifeSupport);
            } else {
                logService = NullLogService.getInstance();
            }
            var jobScheduler = lifeSupport.add(createScheduler());

            lifeSupport.start();

            var nodeStore = NodeStore.of(graphStore);
            var relationshipStore = RelationshipStore.of(graphStore, config.defaultRelationshipType());
            Input input = Neo4jProxy.batchInputFrom(new GraphStoreInput(
                nodeStore,
                relationshipStore,
                config.batchSize()
            ));

            var metaDataPath = Neo4jProxy.metadataStore(databaseLayout);
            var dbExists = Files.exists(metaDataPath) && Files.isReadable(metaDataPath);
            if (dbExists) {
                throw new IllegalArgumentException(formatWithLocale(
                    "The database [%s] already exists. The graph export procedure can only create new databases.",
                    config.dbName()
                ));
            }

            var importer = Neo4jProxy.instantiateBatchImporter(
                BatchImporterFactory.withHighestPriority(),
                databaseLayout,
                fs,
                null, // no external page cache
                PageCacheTracer.NULL,
                importConfig,
                logService,
                ExecutionMonitors.invisible(),
                AdditionalInitialIds.EMPTY,
                databaseConfig,
                RecordFormatSelector.selectForConfig(databaseConfig, logService.getInternalLogProvider()),
                ImportLogic.NO_MONITOR,
                jobScheduler,
                Collector.EMPTY
            );
            importer.doImport(input);

            long importedNodeProperties = nodeStore.propertyCount() * graphStore.nodes().nodeCount();
            long importedRelationshipProperties = relationshipStore.propertyCount() * graphStore.relationshipCount();
            return ImmutableImportedProperties.of(importedNodeProperties, importedRelationshipProperties);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            lifeSupport.shutdown();
        }

        return null;
    }

    @NotNull
    private Configuration getImportConfig(boolean defaultSettingsSuitableForTests) {
        return new Configuration() {
            @Override
            public int maxNumberOfProcessors() {
                return config.writeConcurrency();
            }

            @Override
            public long pageCacheMemory() {
                return defaultSettingsSuitableForTests ? mebiBytes(8) : Configuration.super.pageCacheMemory();
            }

            @Override
            public boolean highIO() {
                return false;
            }
        };
    }

    @ValueClass
    public interface ImportedProperties {

        long nodePropertyCount();

        long relationshipPropertyCount();
    }

    static class NodeStore {

        static final String[] EMPTY_LABELS = new String[0];

        final long nodeCount;

        final HugeIntArray labelCounts;

        final NodeMapping nodeLabels;

        final Map<String, Map<String, NodeProperties>> nodeProperties;

        private final Set<NodeLabel> availableNodeLabels;

        NodeStore(
            long nodeCount,
            HugeIntArray labelCounts,
            NodeMapping nodeLabels,
            Map<String, Map<String, NodeProperties>> nodeProperties
        ) {
            this.nodeCount = nodeCount;
            this.labelCounts = labelCounts;
            this.nodeLabels = nodeLabels;
            this.nodeProperties = nodeProperties;
            this.availableNodeLabels = nodeLabels != null ? nodeLabels.availableNodeLabels() : null;
        }

        boolean hasLabels() {
            return nodeLabels != null;
        }

        boolean hasProperties() {
            return nodeProperties != null;
        }

        int labelCount() {
            return !hasLabels() ? 0 : nodeLabels.availableNodeLabels().size();
        }

        int propertyCount() {
            if (nodeProperties == null) {
                return 0;
            } else {
                return nodeProperties.values().stream().mapToInt(Map::size).sum();
            }
        }

        String[] labels(long nodeId) {
            int labelCount = labelCounts.get(nodeId);
            if (labelCount == 0) {
                return EMPTY_LABELS;
            }
            String[] labels = new String[labelCount];

            int i = 0;
            for (var nodeLabel : availableNodeLabels) {
                if (nodeLabels.hasLabel(nodeId, nodeLabel)) {
                    labels[i++] = nodeLabel.name;
                }
            }

            return labels;
        }

        static NodeStore of(GraphStore graphStore) {
            HugeIntArray labelCounts = null;
            Map<String, Map<String, NodeProperties>> nodeProperties;

            var nodeLabels = graphStore.nodes();

            if (!nodeLabels.containsOnlyAllNodesLabel()) {
                labelCounts = HugeIntArray.newArray(graphStore.nodeCount(), AllocationTracker.empty());
                labelCounts.setAll(i -> {
                    int labelCount = 0;
                    for (var nodeLabel : nodeLabels.availableNodeLabels()) {
                        if (nodeLabels.hasLabel(i, nodeLabel)) {
                            labelCount++;
                        }
                    }
                    return labelCount;
                });
            }

            if (graphStore.nodePropertyKeys().isEmpty()) {
                nodeProperties = null;
            } else {
                nodeProperties = graphStore.nodePropertyKeys().entrySet().stream().collect(Collectors.toMap(
                    entry -> entry.getKey().name,
                    entry -> entry.getValue().stream().collect(Collectors.toMap(
                        propertyKey -> propertyKey,
                        propertyKey -> graphStore.nodePropertyValues(entry.getKey(), propertyKey)
                    ))
                ));
            }
            return new NodeStore(
                graphStore.nodeCount(),
                labelCounts,
                nodeLabels.containsOnlyAllNodesLabel() ? null : nodeLabels,
                nodeProperties
            );
        }
    }

    static class RelationshipStore {

        final long nodeCount;
        final long relationshipCount;

        final Map<String, RelationshipIterator> relationships;

        final Map<String, String> relationshipPropertyKeys;

        final String[] relTypes;

        final String[] propertyKeys;

        RelationshipStore(
            long nodeCount,
            long relationshipCount,
            Map<String, RelationshipIterator> relationships,
            Map<String, String> relationshipPropertyKeys
        ) {
            this.nodeCount = nodeCount;
            this.relationshipCount = relationshipCount;
            this.relationships = relationships;
            this.relationshipPropertyKeys = relationshipPropertyKeys;

            this.relTypes = relationships.keySet().toArray(new String[0]);
            this.propertyKeys = Arrays.stream(relTypes).map(relationshipPropertyKeys::get).toArray(String[]::new);
        }

        int propertyCount() {
            return relationshipPropertyKeys.size();
        }

        RelationshipStore concurrentCopy() {
            return new RelationshipStore(
                nodeCount,
                relationshipCount,
                relationships.entrySet().stream().collect(Collectors.toMap(
                    Map.Entry::getKey,
                    entry -> entry.getValue().concurrentCopy()
                )),
                relationshipPropertyKeys
            );
        }

        static RelationshipStore of(GraphStore graphStore, String defaultRelationshipType) {
            var graphs = graphStore
                .relationshipTypes()
                .stream()
                .flatMap(relType -> {
                    Set<String> relProperties = graphStore.relationshipPropertyKeys(relType);
                    if (relProperties.isEmpty()) {
                        return Stream.of(Tuples.pair(relType, Optional.<String>empty()));
                    } else {
                        return relProperties
                            .stream()
                            .map(propertyKey -> Tuples.pair(relType, Optional.of(propertyKey)));
                    }
                })
                .collect(Collectors.toMap(
                    relTypeAndProperty -> Tuples.pair(
                        relTypeAndProperty.getOne().equals(RelationshipType.ALL_RELATIONSHIPS)
                            ? RelationshipType.of(defaultRelationshipType)
                            : relTypeAndProperty.getOne(),
                        relTypeAndProperty.getTwo()
                    ),
                    relTypeAndProperty -> graphStore.getGraph(relTypeAndProperty.getOne(), relTypeAndProperty.getTwo())
                ));

            Map<String, RelationshipIterator> relationships = graphs.entrySet().stream().collect(Collectors.toMap(
                entry -> entry.getKey().getOne().name,
                Map.Entry::getValue,
                (left, right) -> left
            ));

            var relationshipPropertyKeys = graphs.keySet().stream()
                .filter(pair -> pair.getTwo().isPresent())
                .collect(Collectors.toMap(
                    entry -> entry.getOne().name,
                    entry -> entry.getTwo().get(),
                    (left, right) -> left
                ));

            return new RelationshipStore(
                graphStore.nodeCount(),
                graphStore.relationshipCount(),
                relationships,
                relationshipPropertyKeys
            );
        }
    }

    private static final Validator<Path> DIRECTORY_IS_WRITABLE = value -> {
        try {
            Files.createDirectories(value);
            if (!Files.isDirectory(value)) {
                throw new IllegalArgumentException("'" + value + "' is not a directory");
            }
            if (!Files.isWritable(value)) {
                throw new IllegalArgumentException("Directory '" + value + "' not writable");
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Directory '" + value + "' not writable: ", e);
        }
    };
}
