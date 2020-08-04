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
import org.neo4j.graphalgo.NodeProjections;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.PropertyMappings;
import org.neo4j.graphalgo.RelationshipProjections;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.annotation.Configuration.ConvertWith;
import org.neo4j.graphalgo.annotation.Configuration.Key;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.GraphStoreFactory;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.loading.NativeFactory;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

@ValueClass
@Configuration
@SuppressWarnings("immutables:subtype")
public interface GraphCreateFromStoreConfig extends GraphCreateConfig {

    String NODE_PROJECTION_KEY = "nodeProjection";
    String RELATIONSHIP_PROJECTION_KEY = "relationshipProjection";
    String NODE_PROPERTIES_KEY = "nodeProperties";
    String RELATIONSHIP_PROPERTIES_KEY = "relationshipProperties";

    @Key(NODE_PROJECTION_KEY)
    @ConvertWith("org.neo4j.graphalgo.AbstractNodeProjections#fromObject")
    NodeProjections nodeProjections();

    @Key(RELATIONSHIP_PROJECTION_KEY)
    @ConvertWith("org.neo4j.graphalgo.AbstractRelationshipProjections#fromObject")
    RelationshipProjections relationshipProjections();

    @Value.Default
    @Value.Parameter(false)
    @Configuration.ConvertWith("org.neo4j.graphalgo.AbstractPropertyMappings#fromObject")
    default PropertyMappings nodeProperties() {
        return PropertyMappings.of();
    }

    @Value.Default
    @Value.Parameter(false)
    @Configuration.ConvertWith("org.neo4j.graphalgo.AbstractPropertyMappings#fromObject")
    default PropertyMappings relationshipProperties() {
        return PropertyMappings.of();
    }

    @Configuration.Ignore
    @Override
    default GraphStoreFactory.Supplier graphStoreFactory() {
        return loaderContext -> new NativeFactory(this, loaderContext);
    }

    @Value.Check
    default void validateProjectionsAreNotEmpty() {
        if (nodeProjections().isEmpty()) {
            throw new IllegalArgumentException(
                "The parameter 'nodeProjections' should not be empty. Use '*' to load all nodes."
            );
        }

        if (relationshipProjections().isEmpty()) {
            throw new IllegalArgumentException(
                "The parameter 'relationshipProjections' should not be empty. Use '*' to load all Relationships."
            );
        }
    }

    @Value.Check
    default GraphCreateFromStoreConfig withNormalizedPropertyMappings() {
        PropertyMappings nodeProperties = nodeProperties();
        PropertyMappings relationshipProperties = relationshipProperties();

        if (!nodeProperties.hasMappings() && !relationshipProperties.hasMappings()) {
            return this;
        }

        relationshipProjections().projections().values().forEach(relationshipProjection -> {
            if (relationshipProjection.properties().mappings().size() > 1) {
                throw new IllegalArgumentException(
                    "Implicit graph loading does not allow loading multiple relationship properties per relationship type");
            }
        });

        verifyProperties(
            nodeProperties.stream().map(PropertyMapping::propertyKey).collect(Collectors.toSet()),
            nodeProjections().allProperties(),
            "node"
        );

        verifyProperties(
            relationshipProperties.stream().map(PropertyMapping::propertyKey).collect(Collectors.toSet()),
            relationshipProjections().allProperties(),
            "relationship"
        );

        return ImmutableGraphCreateFromStoreConfig
            .builder()
            .from(this)
            .nodeProjections(nodeProjections().addPropertyMappings(nodeProperties))
            .nodeProperties(PropertyMappings.of())
            .relationshipProjections(relationshipProjections().addPropertyMappings(relationshipProperties))
            .relationshipProperties(PropertyMappings.of())
            .build();
    }

    @Configuration.Ignore
    default void verifyProperties(
        Set<String> propertiesFromMapping,
        Set<String> propertiesFromProjection,
        String type
    ) {
        Set<String> propertyIntersection = new HashSet<>(propertiesFromMapping);
        propertyIntersection.retainAll(propertiesFromProjection);

        if (!propertyIntersection.isEmpty()) {
            throw new IllegalArgumentException(formatWithLocale(
                "Incompatible %s projection and %s property specification. Both specify properties named %s",
                type, type, propertyIntersection
            ));
        }
    }

    @Override
    @Configuration.Ignore
    default <R> R accept(Cases<R> visitor) {
        return visitor.store(this);
    }

    static GraphCreateFromStoreConfig emptyWithName(String userName, String graphName) {
        NodeProjections nodeProjections = NodeProjections.all();
        RelationshipProjections relationshipProjections = RelationshipProjections.all();
        return ImmutableGraphCreateFromStoreConfig.of(
            userName,
            graphName,
            nodeProjections,
            relationshipProjections
        );
    }

    static GraphCreateFromStoreConfig of(
        String userName,
        String graphName,
        Object nodeProjections,
        Object relationshipProjections,
        CypherMapWrapper config
    ) {
        if (nodeProjections != null) {
            config = config.withEntry(NODE_PROJECTION_KEY, nodeProjections);
        }
        if (relationshipProjections != null) {
            config = config.withEntry(RELATIONSHIP_PROJECTION_KEY, relationshipProjections);
        }

        return GraphCreateFromStoreConfigImpl.of(
            graphName,
            userName,
            config
        );
    }

    static GraphCreateFromStoreConfig all(String userName, String graphName) {
        return ImmutableGraphCreateFromStoreConfig.builder()
            .username(userName)
            .graphName(graphName)
            .nodeProjections(NodeProjections.all())
            .relationshipProjections(RelationshipProjections.all())
            .build();
    }

    static GraphCreateFromStoreConfig fromProcedureConfig(String username, CypherMapWrapper config) {
        if (!config.containsKey(NODE_PROJECTION_KEY)) {
            config = config.withEntry(NODE_PROJECTION_KEY, NodeProjections.all());
        }
        if (!config.containsKey(RELATIONSHIP_PROJECTION_KEY)) {
            config = config.withEntry(RELATIONSHIP_PROJECTION_KEY, RelationshipProjections.all());
        }

        return GraphCreateFromStoreConfigImpl.of(
            IMPLICIT_GRAPH_NAME,
            username,
            config
        );
    }
}
