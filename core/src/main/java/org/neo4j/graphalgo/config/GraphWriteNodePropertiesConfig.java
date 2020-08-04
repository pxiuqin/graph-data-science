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
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.List;
import java.util.Optional;

@ValueClass
@Configuration
@SuppressWarnings("immutables:subtype")
public interface GraphWriteNodePropertiesConfig extends GraphExportNodePropertiesConfig {

    // This is necessary because of the initialization order in the generated constructors.
    // If we don't set it, it uses into this.concurrency, which is not initialized yet.
    @Value.Default
    default int writeConcurrency() {
        return concurrency();
    }

    static GraphWriteNodePropertiesConfig of(
        String userName,
        String graphName,
        List<String> nodeProperties,
        List<String> nodeLabels,
        CypherMapWrapper config
    ) {
        return new GraphWriteNodePropertiesConfigImpl(
            Optional.of(graphName),
            nodeProperties,
            nodeLabels,
            userName,
            config
        );
    }

}
