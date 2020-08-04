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
package org.neo4j.graphalgo.core;

import org.immutables.value.Value;
import org.jetbrains.annotations.Nullable;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import static org.neo4j.graphalgo.core.StringSimilarity.jaroWinkler;

public final class ConfigKeyValidation {

    private static final double REQUIRED_SIMILARITY = 0.8;

    private ConfigKeyValidation() {}

    public static void requireOnlyKeysFrom(Collection<String> allowedKeys, Collection<String> configKeys) {
        Collection<String> keys = new HashSet<>(configKeys);
        keys.removeAll(allowedKeys);
        if (keys.isEmpty()) {
            return;
        }
        List<String> suggestions = keys.stream()
            .map(invalid -> {
                List<String> candidates = similarStrings(invalid, allowedKeys);
                candidates.removeAll(configKeys);

                if (candidates.isEmpty()) {
                    return invalid;
                }
                if (candidates.size() == 1) {
                    return String.format(Locale.ENGLISH, "%s (Did you mean [%s]?)", invalid, candidates.get(0));
                }
                return String.format(Locale.ENGLISH, "%s (Did you mean one of [%s]?)", invalid, String.join(", ", candidates));
            })
            .collect(Collectors.toList());

        if (suggestions.size() == 1) {
            throw new IllegalArgumentException(String.format(
                Locale.ENGLISH,
                "Unexpected configuration key: %s",
                suggestions.get(0)
            ));
        }

        throw new IllegalArgumentException(String.format(
            Locale.ENGLISH,
            "Unexpected configuration keys: %s",
            String.join(", ", suggestions)
        ));
    }

    static List<String> similarStrings(CharSequence value, Collection<String> candidates) {
        return candidates.stream()
            .map(candidate -> ImmutableStringAndScore.of(candidate, jaroWinkler(value, candidate)))
            .filter(candidate -> candidate.value() > REQUIRED_SIMILARITY)
            .sorted()
            .map(StringAndScore::string)
            .collect(Collectors.toList());
    }

    @Value.Style(
        allParameters = true,
        builderVisibility = Value.Style.BuilderVisibility.SAME,
        jdkOnly = true,
        overshadowImplementation = true,
        typeAbstract = "*",
        visibility = Value.Style.ImplementationVisibility.PUBLIC
    )
    @Value.Immutable(copy = false, builder = false)
    interface StringAndScore extends Comparable<StringAndScore> {
        String string();

        double value();

        default boolean isBetterThan(@Nullable StringAndScore other) {
            return other == null || value() > other.value();
        }

        @Override
        default int compareTo(StringAndScore other) {
            // ORDER BY score DESC, string ASC
            int result = Double.compare(other.value(), this.value());
            return (result != 0) ? result : this.string().compareTo(other.string());
        }
    }
}
