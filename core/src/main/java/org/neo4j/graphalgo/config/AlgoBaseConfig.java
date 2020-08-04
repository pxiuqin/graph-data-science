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
import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.api.GraphStore;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;

public interface AlgoBaseConfig extends BaseConfig, ConcurrencyConfig {

    String NODE_LABELS_KEY = "nodeLabels";

    @Configuration.Parameter
    Optional<String> graphName();

    @Value.Default
    default List<String> relationshipTypes() {
        return Collections.singletonList(PROJECT_ALL);
    }

    @Configuration.Ignore
    default Collection<RelationshipType> internalRelationshipTypes(GraphStore graphStore) {
        return relationshipTypes().contains(PROJECT_ALL)
            ? graphStore.relationshipTypes()
            : relationshipTypes().stream().map(RelationshipType::of).collect(Collectors.toList());
    }

    @Value.Default
    default List<String> nodeLabels() {
        return Collections.singletonList(PROJECT_ALL);
    }

    @Configuration.Ignore
    default Collection<NodeLabel> nodeLabelIdentifiers(GraphStore graphStore) {
        return nodeLabels().contains(PROJECT_ALL)
            ? graphStore.nodeLabels()
            : nodeLabels().stream().map(NodeLabel::of).collect(Collectors.toList());
    }

    @Configuration.Parameter
    Optional<GraphCreateConfig> implicitCreateConfig();

}
