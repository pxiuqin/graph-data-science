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
package org.neo4j.graphalgo.core.model;

import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.config.BaseConfig;
import org.neo4j.graphalgo.config.TrainConfig;
import org.neo4j.graphalgo.core.utils.TimeUtil;

import java.time.ZonedDateTime;

@ValueClass
public interface Model<DATA, CONFIG extends TrainConfig & BaseConfig> {

    String username();

    String name();

    String algoType();

    DATA data();

    CONFIG trainConfig();

    ZonedDateTime creationTime();

    static <D, C extends TrainConfig & BaseConfig> Model<D, C> of(String username, String name, String algoType, D modelData, C trainConfig) {
        return ImmutableModel.of(username, name, algoType, modelData, trainConfig, TimeUtil.now());
    }
}
