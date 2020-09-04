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
package org.neo4j.graphalgo.beta.pregel;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;

import java.util.function.BiConsumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.arguments;

class CompositeNodeValueTest {

    @ParameterizedTest
    @MethodSource("org.neo4j.graphalgo.beta.pregel.CompositeNodeValueTest#validPropertyTypeAndGetters")
    void testThrowWhenAccessingUnknownProperty(
        ValueType valueType,
        BiConsumer<Pregel.CompositeNodeValue, String> valueConsumer
    ) {
        var schema = new NodeSchemaBuilder().putElement("KEY", valueType).build();
        var nodeValues = Pregel.CompositeNodeValue.of(schema, 10, 4, AllocationTracker.empty());

        var ex = assertThrows(
            IllegalArgumentException.class,
            () -> valueConsumer.accept(nodeValues, "DOES_NOT_EXIST")
        );

        assert (
            ex.getMessage().contains("Property with key DOES_NOT_EXIST does not exist. Available properties are: [KEY]")
        );
    }

    @ParameterizedTest
    @MethodSource("org.neo4j.graphalgo.beta.pregel.CompositeNodeValueTest#invalidPropertyTypeAndGetters")
    void testThrowWhenAccessingPropertyOfWrongType(
        ValueType valueType,
        BiConsumer<Pregel.CompositeNodeValue, String> valueConsumer
    ) {
        var schema = new NodeSchemaBuilder().putElement("KEY", valueType).build();
        var nodeValues = Pregel.CompositeNodeValue.of(schema, 10, 4, AllocationTracker.empty());

        var ex = assertThrows(
            IllegalArgumentException.class,
            () -> valueConsumer.accept(nodeValues, "KEY")
        );

        assert (
            ex.getMessage().contains("Could not cast property KEY")
        );
    }

    static Stream<Arguments> validPropertyTypeAndGetters() {
        BiConsumer<Pregel.CompositeNodeValue, String> longGetter = Pregel.CompositeNodeValue::longProperties;
        BiConsumer<Pregel.CompositeNodeValue, String> doubleGetter = Pregel.CompositeNodeValue::doubleProperties;
        BiConsumer<Pregel.CompositeNodeValue, String> longArrayGetter = Pregel.CompositeNodeValue::longArrayProperties;
        BiConsumer<Pregel.CompositeNodeValue, String> doubleArrayGetter = Pregel.CompositeNodeValue::doubleArrayProperties;
        return Stream.of(
            arguments(ValueType.LONG, longGetter),
            arguments(ValueType.DOUBLE, doubleGetter),
            arguments(ValueType.LONG_ARRAY, longArrayGetter),
            arguments(ValueType.DOUBLE_ARRAY, doubleArrayGetter)
        );
    }

    static Stream<Arguments> invalidPropertyTypeAndGetters() {
        BiConsumer<Pregel.CompositeNodeValue, String> longGetter = Pregel.CompositeNodeValue::longProperties;
        BiConsumer<Pregel.CompositeNodeValue, String> doubleGetter = Pregel.CompositeNodeValue::doubleProperties;

        return Stream.of(
            arguments(ValueType.LONG, doubleGetter),
            arguments(ValueType.DOUBLE, longGetter),
            arguments(ValueType.LONG_ARRAY, longGetter),
            arguments(ValueType.DOUBLE_ARRAY, doubleGetter)
        );
    }
}
