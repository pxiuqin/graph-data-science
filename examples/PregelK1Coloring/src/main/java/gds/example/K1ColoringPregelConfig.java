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
package gds.example;

import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.IterationsConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.Optional;

@Configuration("K1ColoringPregelConfigImpl")
interface K1ColoringPregelConfig extends AlgoBaseConfig, IterationsConfig {
    int DEFAULT_ITERATIONS = 10;

    @Override
    default int maxIterations() {
        return DEFAULT_ITERATIONS;
    }

    static K1ColoringPregelConfig of(
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeCreateConfig,
        String username,
        CypherMapWrapper userInput
    ) {
        return new K1ColoringPregelConfigImpl(graphName, maybeCreateConfig, username, userInput);
    }

}
