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
package org.neo4j.graphalgo.catalog;

import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.config.RandomGraphGeneratorConfig;
import org.neo4j.graphalgo.core.utils.mem.MemoryUsage;

import java.time.ZonedDateTime;
import java.util.Map;

public class GraphInfo {

    public final String graphName;
    public final String database;
    public final String memoryUsage;
    public final long sizeInBytes;
    public final Map<String, Object> nodeProjection;
    public final Map<String, Object> relationshipProjection;
    public final String nodeQuery;
    public final String relationshipQuery;
    public final long nodeCount;
    public final long relationshipCount;
    public final ZonedDateTime creationTime;
    public final ZonedDateTime modificationTime;
    public final Map<String, Object> schema;

    GraphInfo(
        String graphName,
        String database,
        String memoryUsage,
        long sizeInBytes,
        Map<String, Object> nodeProjection,
        Map<String, Object> relationshipProjection,
        String nodeQuery,
        String relationshipQuery,
        long nodeCount,
        long relationshipCount,
        ZonedDateTime creationTime,
        ZonedDateTime modificationTime,
        Map<String, Object> schema
    ) {
        this.graphName = graphName;
        this.database = database;
        this.memoryUsage = memoryUsage;
        this.sizeInBytes = sizeInBytes;
        this.nodeProjection = nodeProjection;
        this.relationshipProjection = relationshipProjection;
        this.nodeQuery = nodeQuery;
        this.relationshipQuery = relationshipQuery;
        this.nodeCount = nodeCount;
        this.relationshipCount = relationshipCount;
        this.creationTime = creationTime;
        this.modificationTime = modificationTime;
        this.schema = schema;
    }

    static GraphInfo of(GraphCreateConfig graphCreateConfig, GraphStore graphStore) {
        var graphName = graphCreateConfig.graphName();
        var creationTime = graphCreateConfig.creationTime();

        var sizeInBytes = MemoryUsage.sizeOf(graphStore);
        var memoryUsage = MemoryUsage.humanReadable(sizeInBytes);

        var configVisitor = new Visitor();
        graphCreateConfig.accept(configVisitor);

        return new GraphInfo(
            graphName,
            graphStore.databaseId().name(),
            memoryUsage,
            sizeInBytes,
            configVisitor.nodeProjection,
            configVisitor.relationshipProjection,
            configVisitor.nodeQuery,
            configVisitor.relationshipQuery,
            graphStore.nodeCount(),
            graphStore.relationshipCount(),
            creationTime,
            graphStore.modificationTime(),
            graphStore.schema().toMap()
        );
    }

    static final class Visitor implements GraphCreateConfig.Visitor {

        String nodeQuery, relationshipQuery = null;
        Map<String, Object> nodeProjection, relationshipProjection = null;

        @Override
        public void visit(GraphCreateFromStoreConfig storeConfig) {
            nodeProjection = storeConfig.nodeProjections().toObject();
            relationshipProjection = storeConfig.relationshipProjections().toObject();
        }

        @Override
        public void visit(GraphCreateFromCypherConfig cypherConfig) {
            nodeQuery = cypherConfig.nodeQuery();
            relationshipQuery = cypherConfig.relationshipQuery();
        }

        @Override
        public void visit(RandomGraphGeneratorConfig randomGraphConfig) {
            nodeProjection = randomGraphConfig.nodeProjections().toObject();
            relationshipProjection = randomGraphConfig.relationshipProjections().toObject();
        }
    }
}
