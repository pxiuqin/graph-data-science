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

package org.neo4j.graphalgo.config;

import org.immutables.value.Value;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.loading.CypherFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROPERTIES_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROPERTIES_KEY;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

@ValueClass
@Configuration
@SuppressWarnings("immutables:subtype")
public interface GraphCreateFromCypherConfig extends GraphCreateConfig {

    List<String> FORBIDDEN_KEYS = Arrays.asList(NODE_PROJECTION_KEY, RELATIONSHIP_PROJECTION_KEY, NODE_PROPERTIES_KEY, RELATIONSHIP_PROPERTIES_KEY);

    String NODE_QUERY_KEY = "nodeQuery";
    String RELATIONSHIP_QUERY_KEY = "relationshipQuery";
    String ALL_NODES_QUERY = "MATCH (n) RETURN id(n) AS id";
    String ALL_RELATIONSHIPS_QUERY = "MATCH (a)-->(b) RETURN id(a) AS source, id(b) AS target";
    String ALL_RELATIONSHIPS_UNDIRECTED_QUERY = "MATCH (a)--(b) RETURN id(a) AS source, id(b) AS target";

    @Configuration.ConvertWith("org.apache.commons.lang3.StringUtils#trimToNull")
    String nodeQuery();

    @Configuration.ConvertWith("org.apache.commons.lang3.StringUtils#trimToNull")
    String relationshipQuery();

    @Value.Default
    default Map<String, Object> parameters() {
        return Collections.emptyMap();
    }

    @Override
    @Value.Default
    @Value.Parameter(false)
    default boolean validateRelationships() {
        return true;
    }

    @Configuration.Ignore
    @Override
    default GraphStoreFactory.Supplier graphStoreFactory() {
        return new GraphStoreFactory.Supplier() {
            @Override
            public GraphStoreFactory<? extends GraphStore, ? extends GraphCreateConfig> get(GraphLoaderContext loaderContext) {
                return new CypherFactory(GraphCreateFromCypherConfig.this, loaderContext);
            }

            @Override
            public GraphStoreFactory<? extends GraphStore, ? extends GraphCreateConfig> getWithDimension(
                GraphLoaderContext loaderContext, GraphDimensions graphDimensions
            ) {
                return new CypherFactory(GraphCreateFromCypherConfig.this, loaderContext, graphDimensions);
            }
        };
    }

    @Override
    @Value.Default
    @Value.Parameter(false)
    default boolean sudo() {
        return true;
    }

    @Override
    @Configuration.Ignore
    default <R> R accept(Cases<R> visitor) {
        return visitor.cypher(this);
    }

    static GraphCreateFromCypherConfig of(
        String userName,
        String graphName,
        String nodeQuery,
        String relationshipQuery,
        CypherMapWrapper config
    ) {
        assertNoProjectionsOrExplicitProperties(config);

        if (nodeQuery != null) {
            config = config.withString(NODE_QUERY_KEY, nodeQuery);
        }
        if (relationshipQuery != null) {
            config = config.withString(RELATIONSHIP_QUERY_KEY, relationshipQuery);
        }
        return new GraphCreateFromCypherConfigImpl(
            graphName,
            userName,
            config
        );
    }

    static GraphCreateFromCypherConfig fromProcedureConfig(String username, CypherMapWrapper config) {
        assertNoProjectionsOrExplicitProperties(config);
        return new GraphCreateFromCypherConfigImpl(
            IMPLICIT_GRAPH_NAME,
            username,
            config
        );
    }

    static void assertNoProjectionsOrExplicitProperties(CypherMapWrapper config) {
        for (String forbiddenKey : FORBIDDEN_KEYS) {
            if (config.containsKey(forbiddenKey)) {
                throw new IllegalArgumentException(formatWithLocale("Invalid key: %s", forbiddenKey));
            }
        }
    }
}
