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
package org.neo4j.graphalgo.nodesim;

import org.immutables.value.Value;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.RelationshipWeightConfig;

import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

//节点相似配置
public interface NodeSimilarityBaseConfig extends AlgoBaseConfig, RelationshipWeightConfig {

    String TOP_K_KEY = "topK";
    int TOP_K_DEFAULT = 10;

    String TOP_N_KEY = "topN";
    int TOP_N_DEFAULT = 0;

    String BOTTOM_K_KEY = "bottomK";
    int BOTTOM_K_DEFAULT = TOP_K_DEFAULT;

    String BOTTOM_N_KEY = "bottomN";
    int BOTTOM_N_DEFAULT = TOP_N_DEFAULT;

    @Value.Default
    @Configuration.DoubleRange(min = 0, max = 1)
    default double similarityCutoff() {
        return 1E-42;
    }

    @Value.Default
    @Configuration.IntegerRange(min = 1)
    default int degreeCutoff() {
        return 1;
    }

    @Value.Default
    @Configuration.Key(TOP_K_KEY)
    @Configuration.IntegerRange(min = 1)
    default int topK() {
        return TOP_K_DEFAULT;
    }

    @Value.Default
    @Configuration.Key(TOP_N_KEY)
    @Configuration.IntegerRange(min = 0)
    default int topN() {
        return TOP_N_DEFAULT;
    }

    @Value.Default
    @Configuration.Key(BOTTOM_K_KEY)
    @Configuration.IntegerRange(min = 1)
    default int bottomK() {
        return BOTTOM_K_DEFAULT;
    }

    @Value.Default
    @Configuration.Key(BOTTOM_N_KEY)
    @Configuration.IntegerRange(min = 0)
    default int bottomN() {
        return BOTTOM_N_DEFAULT;
    }

    @Configuration.Ignore
    @Value.Derived
    default int normalizedK() {
        return bottomK() != BOTTOM_K_DEFAULT
            ? -bottomK()
            : topK();
    }

    @Configuration.Ignore
    @Value.Derived
    default int normalizedN() {
        return bottomN() != BOTTOM_N_DEFAULT
            ? -bottomN()
            : topN();
    }

    @Configuration.Ignore
    @Value.Derived
    default boolean isParallel() {
        return concurrency() > 1;
    }

    @Configuration.Ignore
    @Value.Derived
    default boolean hasTopK() {
        return normalizedK() != 0;
    }

    @Configuration.Ignore
    @Value.Derived
    default boolean hasTopN() {
        return normalizedN() != 0;
    }

    @Configuration.Ignore
    default boolean computeToStream() {
        return false;
    }

    @Configuration.Ignore
    @Value.Derived
    default boolean computeToGraph() {
        return !computeToStream();
    }

    @Value.Check
    default void validate() {
        if (topK() != TOP_K_DEFAULT && bottomK() != BOTTOM_K_DEFAULT) {
            throw new IllegalArgumentException(formatWithLocale(
                "Invalid parameter combination: %s combined with %s",
                TOP_K_KEY,
                BOTTOM_K_KEY
            ));
        }
        if (topN() != TOP_N_DEFAULT && bottomN() != BOTTOM_N_DEFAULT) {
            throw new IllegalArgumentException(formatWithLocale(
                "Invalid parameter combination: %s combined with %s",
                TOP_N_KEY,
                BOTTOM_N_KEY
            ));
        }
    }
}
