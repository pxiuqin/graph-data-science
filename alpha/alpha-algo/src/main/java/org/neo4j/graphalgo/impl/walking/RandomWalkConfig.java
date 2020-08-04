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

package org.neo4j.graphalgo.impl.walking;

import org.immutables.value.Value;
import org.jetbrains.annotations.Nullable;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;

import java.util.Optional;

@ValueClass
@Configuration
@SuppressWarnings("immutables:subtype")
public interface RandomWalkConfig extends AlgoBaseConfig {

    @Value.Default
    default @Nullable Object start() {
        return null;
    }

    @Value.Default
    default long steps() {
        return 10L;
    }

    @Value.Default
    default long walks() {
        return 1L;
    }

    @Value.Default
    default String mode() {
        return "random";
    }

    @Value.Default
    @Configuration.Key(value = "return")
    default double returnKey() {
        return 1.0D;
    }

    @Value.Default
    default double inOut() {
        return 1.0D;
    }

    @Value.Default
    default boolean path() {
        return false;
    }

    static RandomWalkConfig of(
        String username,
        Optional<String> graphName,
        Optional<GraphCreateConfig> maybeImplicitCreate,
        CypherMapWrapper userInput
    ) {
        return new RandomWalkConfigImpl(
            graphName,
            maybeImplicitCreate,
            username,
            userInput
        );
    }

}
