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
package org.neo4j.graphalgo.api.schema;

import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;

import java.util.Map;

import static org.neo4j.graphalgo.compat.MapUtil.map;

@ValueClass
public interface GraphStoreSchema {

    NodeSchema nodeSchema();

    RelationshipSchema relationshipSchema();

    default Map<String, Object> toMap() {
        return map(
            "nodes", nodeSchema().toMap(),
            "relationships", relationshipSchema().toMap()
        );
    }

    static GraphStoreSchema of(NodeSchema nodeSchema, RelationshipSchema relationshipSchema) {
        return ImmutableGraphStoreSchema.builder()
            .nodeSchema(nodeSchema)
            .relationshipSchema(relationshipSchema)
            .build();
    }

    static String forValueType(ValueType vt) {
        return vt.cypherName();
    }
}
