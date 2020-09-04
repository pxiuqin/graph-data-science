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
package org.neo4j.graphalgo;

import org.immutables.value.Value;
import org.jetbrains.annotations.Nullable;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.core.Aggregation;

import java.util.AbstractMap;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

@ValueClass
public abstract class PropertyMapping {

    public static final String PROPERTY_KEY = "property";
    public static final String DEFAULT_VALUE_KEY = "defaultValue";

    /**
     * property key in the result map Graph.nodeProperties(`propertyKey`)
     */
    public abstract @Nullable String propertyKey();

    /**
     * property name in the graph (a:Node {`propertyKey`:xyz})
     */
    @Value.Default
    public @Nullable String neoPropertyKey() {
        return propertyKey();
    }

    @Value.Default
    public DefaultValue defaultValue() {
        return DefaultValue.DEFAULT;
    }

    @Value.Default
    public Aggregation aggregation() {
        return Aggregation.DEFAULT;
    }

    @Value.Check
    public void validateProperties() {
        if (neoPropertyKey().equals(PROJECT_ALL) && aggregation() != Aggregation.COUNT) {
            throw new IllegalArgumentException("A '*' property key can only be used in combination with count aggregation.");
        }
    }

    public static PropertyMapping fromObject(String propertyKey, Object stringOrMap) {
        if (stringOrMap instanceof String) {
            String neoPropertyKey = (String) stringOrMap;
            return fromObject(
                propertyKey,
                Collections.singletonMap(
                    PROPERTY_KEY,
                    neoPropertyKey
                )
            );
        } else if (stringOrMap instanceof Map) {
            Map<String, Object> relPropertyMap = (Map) stringOrMap;
            Object propertyNameValue = relPropertyMap.getOrDefault(PROPERTY_KEY, propertyKey);
            if (!(propertyNameValue instanceof String)) {
                throw new IllegalArgumentException(formatWithLocale(
                    "Expected the value of '%s' to be of type String, but was '%s'.",
                    PROPERTY_KEY, propertyNameValue.getClass().getSimpleName()
                ));
            }
            String neoPropertyKey = (String) propertyNameValue;

            final Object aggregationValue = relPropertyMap.get(RelationshipProjection.AGGREGATION_KEY);
            Aggregation aggregation;
            if (aggregationValue == null) {
                aggregation = Aggregation.DEFAULT;
            } else if (aggregationValue instanceof String) {
                aggregation = Aggregation.lookup(((String) aggregationValue).toUpperCase(Locale.ENGLISH));
            } else {
                throw new IllegalStateException(formatWithLocale(
                    "Expected the value of '%s' to be of type String, but was '%s'",
                    RelationshipProjection.AGGREGATION_KEY, aggregationValue.getClass().getSimpleName()
                ));
            }

            return PropertyMapping.of(
                propertyKey,
                neoPropertyKey,
                relPropertyMap.get(DEFAULT_VALUE_KEY),
                aggregation
            );
        } else {
            throw new IllegalStateException(formatWithLocale(
                "Expected stringOrMap to be of type String or Map, but got %s",
                stringOrMap.getClass().getSimpleName()
            ));
        }
    }

    public boolean hasValidName() {
        String key = neoPropertyKey();
        return key != null && !key.isEmpty();
    }

    public boolean exists() {
        return false;
    }

    public Map.Entry<String, Object> toObject(boolean includeAggregation) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put(PROPERTY_KEY, neoPropertyKey());
        value.put(DEFAULT_VALUE_KEY, defaultValue().getObject());
        if (includeAggregation) {
            value.put(RelationshipProjection.AGGREGATION_KEY, aggregation().name());
        }
        return new AbstractMap.SimpleImmutableEntry<>(propertyKey(), value);
    }

    public PropertyMapping setNonDefaultAggregation(Aggregation aggregation) {
        if (aggregation == Aggregation.DEFAULT || aggregation() != Aggregation.DEFAULT) {
            return this;
        }
        return ((ImmutablePropertyMapping) this).withAggregation(aggregation);
    }

    public static PropertyMapping of(String propertyKey) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(propertyKey)
            .build();
    }

    /**
     * Creates a PropertyMapping. The given property key is also used for internal reference.
     */
    public static PropertyMapping of(String neoPropertyKey, Object defaultValue) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(neoPropertyKey)
            .defaultValue(DefaultValue.of(defaultValue))
            .build();
    }

    public static PropertyMapping of(String propertyKey, String neoPropertyKey, Object defaultValue) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(propertyKey)
            .neoPropertyKey(neoPropertyKey)
            .defaultValue(DefaultValue.of(defaultValue))
            .build();
    }

    public static PropertyMapping of(
        String propertyKey,
        Object defaultValue,
        Aggregation aggregation
    ) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(propertyKey)
            .defaultValue(DefaultValue.of(defaultValue))
            .aggregation(aggregation)
            .build();
    }

    public static PropertyMapping of(
        String propertyKey,
        Aggregation aggregation
    ) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(propertyKey)
            .aggregation(aggregation)
            .build();
    }

    public static PropertyMapping of(
        String propertyKey,
        String neoPropertyKey,
        Aggregation aggregation
    ) {
        return ImmutablePropertyMapping
            .builder()
            .propertyKey(propertyKey)
            .neoPropertyKey(neoPropertyKey)
            .aggregation(aggregation)
            .build();
    }

    public static PropertyMapping of(
        String propertyKey,
        String neoPropertyKey,
        Object defaultValue,
        Aggregation aggregation
    ) {
        return ImmutablePropertyMapping.of(propertyKey, neoPropertyKey, DefaultValue.of(defaultValue), aggregation);
    }
}
