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
package org.neo4j.graphalgo.gdl;

import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.api.IdMapping;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.Relationships;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.loading.CSRGraphStore;
import org.neo4j.graphalgo.core.loading.HugeGraphUtil;
import org.neo4j.graphalgo.core.loading.IdMap;
import org.neo4j.graphalgo.core.loading.IdsAndProperties;
import org.neo4j.graphalgo.core.loading.NodePropertiesBuilder;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.extension.GdlSupportExtension;
import org.neo4j.internal.kernel.api.security.AuthSubject;
import org.neo4j.kernel.database.NamedDatabaseId;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.logging.Log;
import org.neo4j.logging.NullLog;
import org.s1ck.gdl.GDLHandler;
import org.s1ck.gdl.model.Element;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public final class GdlFactory extends GraphStoreFactory<CSRGraphStore, GraphCreateFromGdlConfig> {

    private final GDLHandler gdlHandler;
    private final NamedDatabaseId databaseId;

    public static GdlFactory of(String gdlGraph) {
        return of(gdlGraph, GdlSupportExtension.DATABASE_ID);
    }

    public static GdlFactory of(String gdlGraph, NamedDatabaseId namedDatabaseId) {
        return of(AuthSubject.ANONYMOUS.username(), namedDatabaseId, "graph", gdlGraph);
    }

    public static GdlFactory of(String username, NamedDatabaseId namedDatabaseId, String graphName, String gdlGraph) {
        return of(
            ImmutableGraphCreateFromGdlConfig.builder()
                .username(username)
                .graphName(graphName)
                .gdlGraph(gdlGraph)
                .build(),
            namedDatabaseId
        );
    }

    public static GdlFactory of(GraphCreateFromGdlConfig config, NamedDatabaseId databaseId) {
        var gdlHandler = new GDLHandler.Builder()
            .setDefaultVertexLabel(NodeLabel.ALL_NODES.name)
            .setDefaultEdgeLabel(RelationshipType.ALL_RELATIONSHIPS.name)
            .buildFromString(config.gdlGraph());

        var graphDimensions = GraphDimensionsGdlReader.of(gdlHandler);

        return new GdlFactory(gdlHandler, config, graphDimensions, databaseId);
    }

    private GdlFactory(
        GDLHandler gdlHandler,
        GraphCreateFromGdlConfig graphCreateConfig,
        GraphDimensions graphDimensions,
        NamedDatabaseId databaseId
    ) {
        super(graphCreateConfig, NO_API_CONTEXT, graphDimensions);
        this.gdlHandler = gdlHandler;
        this.databaseId = databaseId;
    }

    public long nodeId(String variable) {
        return gdlHandler.getVertexCache().get(variable).getId();
    }

    @Override
    public MemoryEstimation memoryEstimation() {
        return MemoryEstimations.empty();
    }

    @Override
    protected ProgressLogger initProgressLogger() {
        return ProgressLogger.NULL_LOGGER;
    }

    @Override
    public ImportResult<CSRGraphStore> build() {
        var nodes = loadNodes();
        var relationships = loadRelationships(nodes.idMap());
        var topologies = relationships.entrySet().stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> entry.getValue().getTwo().topology()
            ));
        var properties = relationships.entrySet().stream()
            .filter(entry -> entry.getValue().getOne().isPresent())
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> Map.of(entry.getValue().getOne().get(), entry.getValue().getTwo().properties().get())
            ));
        CSRGraphStore graphStore = CSRGraphStore.of(
            databaseId,
            nodes.idMap(),
            nodes.properties(),
            topologies,
            properties,
            1,
            loadingContext.tracker()
        );
        return ImportResult.of(dimensions, graphStore);
    }

    private IdsAndProperties loadNodes() {
        var idMapBuilder = HugeGraphUtil.idMapBuilder(
            dimensions.highestNeoId(),
            loadingContext.executor(),
            loadingContext.tracker()
        );

        gdlHandler.getVertices().forEach(vertex -> idMapBuilder.addNode(
            vertex.getId(),
            vertex.getLabels().stream()
                .map(NodeLabel::of)
                .filter(nodeLabel -> !nodeLabel.equals(NodeLabel.ALL_NODES))
                .toArray(NodeLabel[]::new)
        ));

        var idMap = idMapBuilder.build();

        return IdsAndProperties.of(idMap, loadNodeProperties(idMap));
    }

    private Map<NodeLabel, Map<PropertyMapping, NodeProperties>> loadNodeProperties(IdMapping idMap) {
        var propertyKeysByLabel = new HashMap<NodeLabel, Set<PropertyMapping>>();
        var propertyBuilders = new HashMap<PropertyMapping, NodePropertiesBuilder>();

        gdlHandler.getVertices().forEach(vertex -> vertex
            .getProperties()
            .forEach((propertyKey, propertyValue) -> {
                vertex.getLabels().stream()
                    .map(NodeLabel::of)
                    .forEach(nodeLabel -> propertyKeysByLabel
                        .computeIfAbsent(nodeLabel, (ignore) -> new HashSet<>())
                        .add(PropertyMapping.of(propertyKey))
                    );
                propertyBuilders.computeIfAbsent(PropertyMapping.of(propertyKey), (key) ->
                    NodePropertiesBuilder.of(
                        dimensions.nodeCount(),
                        ValueType.DOUBLE,
                        loadingContext.tracker(),
                        PropertyMapping.DEFAULT_FALLBACK_VALUE
                    )).set(idMap.toMappedNodeId(vertex.getId()), gdsValue(vertex, propertyKey, propertyValue));
            }));

        var nodeProperties = propertyBuilders
            .entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getKey, entry -> entry.getValue().build()));

        return propertyKeysByLabel.entrySet().stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> entry.getValue().stream().collect(Collectors.toMap(
                    propertyKey -> propertyKey,
                    nodeProperties::get
                ))
            ));
    }

    private Map<RelationshipType, Pair<Optional<String>, Relationships>> loadRelationships(IdMap nodes) {
        var propertyKeysByRelType = new HashMap<String, Optional<String>>();

        gdlHandler.getEdges()
            .forEach(edge -> propertyKeysByRelType
                .putIfAbsent(edge.getLabel(), edge.getProperties().keySet().stream().findFirst()));

        var relTypeImporters = propertyKeysByRelType.entrySet().stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                relTypeAndProperty -> HugeGraphUtil.createRelImporter(
                    nodes,
                    graphCreateConfig.orientation(),
                    relTypeAndProperty.getValue().isPresent(),
                    Aggregation.NONE,
                    loadingContext.executor(),
                    loadingContext.tracker()
                )
            ));

        gdlHandler.getEdges()
            .forEach(edge -> {
                var relationshipsBuilder = relTypeImporters.get(edge.getLabel());
                var maybePropertyKey = propertyKeysByRelType.get(edge.getLabel());
                if (maybePropertyKey.isPresent()) {
                    relationshipsBuilder.add(
                        edge.getSourceVertexId(),
                        edge.getTargetVertexId(),
                        gdsValue(edge, maybePropertyKey.get(), edge.getProperties().get(maybePropertyKey.get()))
                    );
                } else {
                    relationshipsBuilder.add(edge.getSourceVertexId(), edge.getTargetVertexId());
                }
            });

        // Add fake relationship type since we do not
        // support GraphStores with zero relationships.
        if (relTypeImporters.isEmpty()) {
            relTypeImporters.put(RelationshipType.ALL_RELATIONSHIPS.name, HugeGraphUtil.createRelImporter(
                nodes,
                graphCreateConfig.orientation(),
                false,
                Aggregation.NONE,
                loadingContext.executor(),
                loadingContext.tracker()
            ));
            propertyKeysByRelType.put(RelationshipType.ALL_RELATIONSHIPS.name, Optional.empty());
        }

        return relTypeImporters.entrySet().stream().collect(Collectors.toMap(
            entry -> RelationshipType.of(entry.getKey()),
            entry -> Tuples.pair(propertyKeysByRelType.get(entry.getKey()), entry.getValue().build())
        ));
    }

    private double gdsValue(Element element, String propertyKey, Object gdlValue) {
        if (gdlValue instanceof Number) {
            return ((Number) gdlValue).doubleValue();
        } else if (gdlValue instanceof String && gdlValue.equals("NaN")) {
            return Double.NaN;
        } else {
            throw new IllegalArgumentException(String.format(
                Locale.ENGLISH,
                "%s property '%s' must be of type Number, but was %s for %s.",
                element.getClass().getTypeName(),
                propertyKey,
                gdlValue.getClass(),
                element
            ));
        }
    }

    private static final GraphLoaderContext NO_API_CONTEXT = new GraphLoaderContext() {
        @Override
        public GraphDatabaseAPI api() {
            return null;
        }

        @Override
        public Log log() {
            return NullLog.getInstance();
        }
    };

    private static final class GraphDimensionsGdlReader {

        static GraphDimensions of(GDLHandler gdlHandler) {
            var nodeCount = gdlHandler.getVertices().size();
            var relCount = gdlHandler.getEdges().size();

            return ImmutableGraphDimensions.builder()
                .nodeCount(nodeCount)
                .maxRelCount(relCount)
                .build();
        }
    }
}
