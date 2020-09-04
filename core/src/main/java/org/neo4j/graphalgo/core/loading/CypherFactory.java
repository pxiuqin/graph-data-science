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

import org.immutables.value.Value;
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.NodeProjection;
import org.neo4j.graphalgo.NodeProjections;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipProjections;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.GraphDimensionsCypherReader;
import org.neo4j.graphalgo.core.ImmutableGraphDimensions;
import org.neo4j.graphalgo.core.SecureTransaction;
import org.neo4j.graphalgo.core.utils.BatchingProgressLogger;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphdb.Transaction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;
import static org.neo4j.graphalgo.core.loading.CypherRecordLoader.QueryType.NODE;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;
import static org.neo4j.internal.kernel.api.security.AccessMode.Static.READ;
import static org.neo4j.kernel.api.StatementConstants.NO_SUCH_PROPERTY_KEY;

public class CypherFactory extends GraphStoreFactory<CSRGraphStore, GraphCreateFromCypherConfig> {

    private final GraphCreateFromCypherConfig cypherConfig;
    private EstimationResult nodeEstimation;
    private EstimationResult relationshipEstimation;

    public CypherFactory(
        GraphCreateFromCypherConfig graphCreateConfig,
        GraphLoaderContext loadingContext
    ) {
        this(
            graphCreateConfig,
            loadingContext,
            new GraphDimensionsCypherReader(
                loadingContext.transaction().withRestrictedAccess(READ),
                graphCreateConfig
            ).call()
        );
    }

    public CypherFactory(
        GraphCreateFromCypherConfig graphCreateConfig,
        GraphLoaderContext loadingContext,
        GraphDimensions graphDimensions
    ) {
        super(graphCreateConfig, loadingContext, graphDimensions);
        this.cypherConfig = getCypherConfig(graphCreateConfig).orElseThrow(() -> new IllegalArgumentException(
            "Expected GraphCreateConfig to be a cypher config."));
    }

    public final MemoryEstimation memoryEstimation() {
        if (cypherConfig.isFictitiousLoading()) {
            nodeEstimation = ImmutableEstimationResult.of(cypherConfig.nodeCount(), 0);
            relationshipEstimation = ImmutableEstimationResult.of(cypherConfig.relationshipCount(), 0);
        }

        var nodeProjection = NodeProjection
            .builder()
            .label(PROJECT_ALL)
            .addAllProperties(getNodeEstimation().propertyMappings())
            .build();

        var nodeProjections = NodeProjections.single(
            NodeLabel.ALL_NODES,
            nodeProjection
        );

        var relationshipProjection = RelationshipProjection
            .builder()
            .type(PROJECT_ALL)
            .addAllProperties(getRelationshipEstimation().propertyMappings())
            .build();

        var relationshipProjections = RelationshipProjections.single(
            RelationshipType.ALL_RELATIONSHIPS,
            relationshipProjection
        );

        return NativeFactory.getMemoryEstimation(nodeProjections, relationshipProjections);
    }

    @Override
    public GraphDimensions estimationDimensions() {
        return ImmutableGraphDimensions.builder()
            .from(dimensions)
            .highestNeoId(getNodeEstimation().estimatedRows())
            .nodeCount(getNodeEstimation().estimatedRows())
            .maxRelCount(getRelationshipEstimation().estimatedRows())
            .build();
    }

    @Override
    public ImportResult build() {
        // Temporarily override the security context to enforce read-only access during load
        return readOnlyTransaction().apply((tx, ktx) -> {
            BatchLoadResult nodeCount = new CountingCypherRecordLoader(
                nodeQuery(),
                NODE,
                loadingContext.api(),
                cypherConfig,
                loadingContext
            ).load(tx);

            CypherNodeLoader.LoadResult nodes = new CypherNodeLoader(
                nodeQuery(),
                nodeCount.rows(),
                loadingContext.api(),
                cypherConfig,
                loadingContext,
                dimensions
            ).load(tx);

            RelationshipImportResult relationships = loadRelationships(
                relationshipQuery(),
                nodes.idsAndProperties(),
                nodes.dimensions(),
                tx
            );

            GraphStore graphStore = createGraphStore(
                nodes.idsAndProperties(),
                relationships,
                loadingContext.tracker(),
                relationships.dimensions()
            );

            progressLogger.logMessage(loadingContext.tracker());
            return ImportResult.of(relationships.dimensions(), graphStore);
        });
    }

    @Override
    protected ProgressLogger initProgressLogger() {
        return new BatchingProgressLogger(
            loadingContext.log(),
            dimensions.nodeCount() + dimensions.maxRelCount(),
            TASK_LOADING,
            graphCreateConfig.readConcurrency()
        );
    }

    private String nodeQuery() {
        return getCypherConfig(graphCreateConfig)
            .orElseThrow(() -> new IllegalArgumentException("Missing node query"))
            .nodeQuery();
    }

    private String relationshipQuery() {
        return getCypherConfig(graphCreateConfig)
            .orElseThrow(() -> new IllegalArgumentException("Missing relationship query"))
            .relationshipQuery();
    }

    private static Optional<GraphCreateFromCypherConfig> getCypherConfig(GraphCreateConfig config) {
        if (config instanceof GraphCreateFromCypherConfig) {
            return Optional.of((GraphCreateFromCypherConfig) config);
        }
        return Optional.empty();
    }

    private RelationshipImportResult loadRelationships(
        String relationshipQuery,
        IdsAndProperties idsAndProperties,
        GraphDimensions nodeLoadDimensions,
        Transaction transaction
    ) {
        CypherRelationshipLoader relationshipLoader = new CypherRelationshipLoader(
            relationshipQuery,
            idsAndProperties.idMap(),
            loadingContext.api(),
            cypherConfig,
            loadingContext,
            nodeLoadDimensions
        );

        CypherRelationshipLoader.LoadResult result = relationshipLoader.load(transaction);

        return RelationshipImportResult.of(
            relationshipLoader.allBuilders(),
            result.relationshipCounts(),
            result.dimensions()
        );
    }

    private SecureTransaction readOnlyTransaction() {
        return loadingContext.transaction().withRestrictedAccess(READ);
    }

    private EstimationResult getNodeEstimation() {
        if (nodeEstimation == null) {
            nodeEstimation = runEstimationQuery(
                nodeQuery(),
                NodeRowVisitor.RESERVED_COLUMNS
            );
        }
        return nodeEstimation;
    }

    private EstimationResult getRelationshipEstimation() {
        if (relationshipEstimation == null) {
            relationshipEstimation = runEstimationQuery(
                relationshipQuery(),
                RelationshipRowVisitor.RESERVED_COLUMNS
            );
        }
        return relationshipEstimation;
    }

    private EstimationResult runEstimationQuery(String query, Collection<String> reservedColumns) {
        return readOnlyTransaction().apply((tx, ktx) -> {
            var explainQuery = formatWithLocale("EXPLAIN %s", query);
            try (var result = tx.execute(explainQuery)) {
                var estimatedRows = (Number) result.getExecutionPlanDescription().getArguments().get("EstimatedRows");

                var propertyColumns = new ArrayList<>(result.columns());
                propertyColumns.removeAll(reservedColumns);

                return ImmutableEstimationResult.of(estimatedRows.longValue(), propertyColumns.size());
            }
        });
    }

    @ValueClass
    interface EstimationResult {
        long estimatedRows();
        long propertyCount();

        @Value.Derived
        default Map<String, Integer> propertyTokens() {
            return LongStream
                .range(0, propertyCount())
                .boxed()
                .collect(Collectors.toMap(
                    Object::toString,
                    property -> NO_SUCH_PROPERTY_KEY
                ));
        }
        @Value.Derived
        default Collection<PropertyMapping> propertyMappings() {
            return LongStream
                .range(0, propertyCount())
                .mapToObj(property -> PropertyMapping.of(Long.toString(property), DefaultValue.DEFAULT))
                .collect(Collectors.toList());
        }

    }
}
