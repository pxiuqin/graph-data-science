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

import com.carrotsearch.hppc.ObjectLongHashMap;
import com.carrotsearch.hppc.ObjectLongMap;
import org.jetbrains.annotations.NotNull;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.api.IdMapping;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.utils.ProgressLogger;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;


public final class ScanningRelationshipsImporter extends ScanningRecordsImporter<RelationshipReference, ObjectLongMap<RelationshipType>> {

    private final GraphCreateConfig graphCreateConfig;
    private final GraphLoaderContext loadingContext;
    private final ProgressLogger progressLogger;

    private final IdMapping idMap;
    private final Map<RelationshipType, RelationshipsBuilder> allBuilders;
    private final Map<RelationshipType, LongAdder> allRelationshipCounters;

    public ScanningRelationshipsImporter(
        GraphCreateConfig graphCreateConfig,
        GraphLoaderContext loadingContext,
        GraphDimensions dimensions,
        ProgressLogger progressLogger,
        IdMapping idMap,
        Map<RelationshipType, RelationshipsBuilder> allBuilders,
        int concurrency
    ) {
        super(
            USE_KERNEL_CURSORS.get() ? RelationshipScanCursorBasedScanner.FACTORY : RelationshipRecordBasedScanner.FACTORY,
            "Relationship",
            loadingContext,
            dimensions,
            concurrency
        );
        this.graphCreateConfig = graphCreateConfig;
        this.loadingContext = loadingContext;
        this.progressLogger = progressLogger;
        this.idMap = idMap;
        this.allBuilders = allBuilders;
        this.allRelationshipCounters = new HashMap<>();
    }

    @Override
    public InternalImporter.CreateScanner creator(
        final long nodeCount,
        final ImportSizing sizing,
        final StoreScanner<RelationshipReference> scanner
    ) {

        int pageSize = sizing.pageSize();
        int numberOfPages = sizing.numberOfPages();

        List<SingleTypeRelationshipImporter.Builder> importerBuilders = allBuilders
                .entrySet()
                .stream()
                .map(entry -> {
                    var relationshipType = entry.getKey();
                    var relationshipsBuilder = entry.getValue();
                    return createImporterBuilder(
                        pageSize,
                        numberOfPages,
                        relationshipType,
                        relationshipsBuilder.projection(),
                        relationshipsBuilder
                    );
                })
                .collect(Collectors.toList());

        for (SingleTypeRelationshipImporter.Builder importerBuilder : importerBuilders) {
            allRelationshipCounters.put(importerBuilder.relationshipType(), importerBuilder.relationshipCounter());
        }

        return RelationshipsScanner.of(
            loadingContext,
            progressLogger,
            idMap,
            scanner,
            importerBuilders
        );
    }

    private SingleTypeRelationshipImporter.Builder createImporterBuilder(
            int pageSize,
            int numberOfPages,
            RelationshipType relationshipType,
            RelationshipProjection projection,
            @NotNull RelationshipsBuilder relationshipsBuilder
    ) {
        List<PropertyMapping> propertyMappings = projection.properties().mappings();
        int[] propertyKeyIds = propertyMappings
            .stream()
            .mapToInt(mapping -> dimensions.relationshipPropertyTokens().get(mapping.neoPropertyKey())).toArray();

        double[] defaultValues = propertyMappings.stream().mapToDouble(propertyMapping -> propertyMapping.defaultValue().doubleValue()).toArray();
        Aggregation[] aggregations = propertyMappings.stream()
            .map(PropertyMapping::aggregation)
            .map(Aggregation::resolve)
            .toArray(Aggregation[]::new);

        if (propertyMappings.isEmpty()) {
            aggregations = new Aggregation[]{ Aggregation.resolve(projection.aggregation()) };
        }

        LongAdder relationshipCounter = new LongAdder();
        AdjacencyBuilder adjacencyBuilder = AdjacencyBuilder.compressing(
            relationshipsBuilder,
            numberOfPages,
            pageSize,
            tracker,
            relationshipCounter,
            propertyKeyIds,
            defaultValues,
            aggregations
        );

        RelationshipImporter importer = new RelationshipImporter(loadingContext.tracker(), adjacencyBuilder);
        int typeId = dimensions.relationshipTypeTokenMapping().get(relationshipType);
        return new SingleTypeRelationshipImporter.Builder(
            relationshipType,
            projection,
            adjacencyBuilder.supportsProperties(),
            typeId,
            importer,
            relationshipCounter,
            graphCreateConfig.validateRelationships()
        );
    }

    @Override
    public ObjectLongMap<RelationshipType> build() {
        ObjectLongMap<RelationshipType> relationshipCounters = new ObjectLongHashMap<>(allRelationshipCounters.size());
        allRelationshipCounters.forEach((relationshipType, counter) -> relationshipCounters.put(relationshipType, counter.sum()));
        return relationshipCounters;
    }
}
