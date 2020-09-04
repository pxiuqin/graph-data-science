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
package org.neo4j.graphalgo.core.loading;

import com.carrotsearch.hppc.ObjectLongMap;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.NodeProjections;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.PropertyMappings;
import org.neo4j.graphalgo.RelationshipProjections;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.GraphDimensionsStoreReader;
import org.neo4j.graphalgo.core.huge.HugeGraph;
import org.neo4j.graphalgo.core.huge.TransientAdjacencyList;
import org.neo4j.graphalgo.core.huge.TransientAdjacencyOffsets;
import org.neo4j.graphalgo.core.loading.nodeproperties.NodePropertiesFromStoreBuilder;
import org.neo4j.graphalgo.core.utils.BatchingProgressLogger;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;

import java.util.Map;
import java.util.stream.Collectors;

import static org.neo4j.graphalgo.core.GraphDimensionsValidation.validate;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public final class NativeFactory extends GraphStoreFactory<CSRGraphStore, GraphCreateFromStoreConfig> {

    private final GraphCreateFromStoreConfig storeConfig;

    public NativeFactory(
        GraphCreateFromStoreConfig graphCreateConfig,
        GraphLoaderContext loadingContext
    ) {
        this(
            graphCreateConfig,
            loadingContext,
            new GraphDimensionsStoreReader(loadingContext.transaction(), graphCreateConfig).call()
        );
    }

    public NativeFactory(
        GraphCreateFromStoreConfig graphCreateConfig,
        GraphLoaderContext loadingContext,
        GraphDimensions graphDimensions
    ) {
        super(graphCreateConfig, loadingContext, graphDimensions);
        this.storeConfig = graphCreateConfig;
    }

    @Override
    public MemoryEstimation memoryEstimation() {
        return getMemoryEstimation(storeConfig.nodeProjections(), storeConfig.relationshipProjections());
    }

    public static MemoryEstimation getMemoryEstimation(NodeProjections nodeProjections, RelationshipProjections relationshipProjections) {
        MemoryEstimations.Builder builder = MemoryEstimations.builder(HugeGraph.class);

        // node information
        builder.add("nodeIdMap", IdMap.memoryEstimation());

        // nodeProperties
        nodeProjections.allProperties()
            .forEach(property -> builder.add(property, NodePropertiesFromStoreBuilder.memoryEstimation()));

        // relationships
        relationshipProjections.projections().forEach((relationshipType, relationshipProjection) -> {

            boolean undirected = relationshipProjection.orientation() == Orientation.UNDIRECTED;

            // adjacency list
            builder.add(
                formatWithLocale("adjacency list for '%s'", relationshipType),
                TransientAdjacencyList.compressedMemoryEstimation(relationshipType, undirected)
            );
            builder.add(
                formatWithLocale("adjacency offsets for '%s'", relationshipType),
                TransientAdjacencyOffsets.memoryEstimation()
            );
            // all properties per projection
            relationshipProjection.properties().mappings().forEach(resolvedPropertyMapping -> {
                builder.add(
                    formatWithLocale("property '%s.%s", relationshipType, resolvedPropertyMapping.propertyKey()),
                    TransientAdjacencyList.uncompressedMemoryEstimation(relationshipType, undirected)
                );
                builder.add(
                    formatWithLocale("property offset '%s.%s", relationshipType, resolvedPropertyMapping.propertyKey()),
                    TransientAdjacencyOffsets.memoryEstimation()
                );
            });
        });

        return builder.build();
    }

    @Override
    protected ProgressLogger initProgressLogger() {
        long relationshipCount = graphCreateConfig
            .relationshipProjections()
            .projections()
            .entrySet()
            .stream()
            .map(entry -> {
                Long relCount = entry.getKey().name.equals("*")
                     ? dimensions.relationshipCounts().values().stream().reduce(Long::sum).orElse(0L)
                     : dimensions.relationshipCounts().getOrDefault(entry.getKey(), 0L);

                return entry.getValue().orientation() == Orientation.UNDIRECTED
                    ? relCount * 2
                    : relCount;
            }).mapToLong(Long::longValue).sum();

        return new BatchingProgressLogger(
            loadingContext.log(),
            dimensions.nodeCount() + relationshipCount,
            TASK_LOADING,
            graphCreateConfig.readConcurrency()
        );
    }

    @Override
    public ImportResult build() {
        validate(dimensions, storeConfig);

        int concurrency = graphCreateConfig.readConcurrency();
        AllocationTracker tracker = loadingContext.tracker();
        IdsAndProperties nodes = loadNodes(concurrency);
        RelationshipImportResult relationships = loadRelationships(tracker, nodes, concurrency);
        GraphStore graphStore = createGraphStore(nodes, relationships, tracker, dimensions);
        progressLogger.logMessage(tracker);

        return ImportResult.of(dimensions, graphStore);
    }

    private IdsAndProperties loadNodes(int concurrency) {
        Map<NodeLabel, PropertyMappings> propertyMappingsByNodeLabel = graphCreateConfig
            .nodeProjections()
            .projections()
            .entrySet()
            .stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> entry.getValue().properties()
            ));

        return new ScanningNodesImporter(
            graphCreateConfig,
            loadingContext,
            dimensions,
            progressLogger,
            concurrency,
            propertyMappingsByNodeLabel
        ).call(loadingContext.log());
    }

    private RelationshipImportResult loadRelationships(
        AllocationTracker tracker,
        IdsAndProperties idsAndProperties,
        int concurrency
    ) {
        var pageSize = ImportSizing.of(concurrency, dimensions.nodeCount()).pageSize();
        Map<RelationshipType, RelationshipsBuilder> allBuilders = graphCreateConfig
            .relationshipProjections()
            .projections()
            .entrySet()
            .stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                projectionEntry -> new RelationshipsBuilder(
                    projectionEntry.getValue(),
                    TransientAdjacencyListBuilder.builderFactory(tracker),
                    TransientAdjacencyOffsets.forPageSize(pageSize)
                )
            ));

        ObjectLongMap<RelationshipType> relationshipCounts = new ScanningRelationshipsImporter(
            graphCreateConfig,
            loadingContext,
            dimensions,
            progressLogger,
            idsAndProperties.idMap,
            allBuilders,
            concurrency
        ).call(loadingContext.log());

        return RelationshipImportResult.of(allBuilders, relationshipCounts, dimensions);
    }
}
