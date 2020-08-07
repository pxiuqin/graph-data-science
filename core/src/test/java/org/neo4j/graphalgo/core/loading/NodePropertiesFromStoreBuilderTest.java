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
package org.neo4j.graphalgo.core.loading;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.core.loading.nodeproperties.NodePropertiesFromStoreBuilder;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.values.storable.Values;

import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.concurrent.Executors;
import java.util.concurrent.Phaser;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class NodePropertiesFromStoreBuilderTest {

    @Test
    void testEmptyDoubleProperties() {
        var properties = NodePropertiesFromStoreBuilder.of(
            100_000,
            AllocationTracker.EMPTY,
            DefaultValue.of(42.0D)
        ).build();

        assertEquals(0L, properties.size());
        assertEquals(OptionalDouble.empty(), properties.getMaxDoublePropertyValue());
        assertEquals(42.0, properties.getDouble(0));
    }

    @Test
    void testEmptyLongProperties() {
        var properties = NodePropertiesFromStoreBuilder.of(
            100_000,
            AllocationTracker.EMPTY,
            DefaultValue.of(42L)
        ).build();

        assertEquals(0L, properties.size());
        assertEquals(OptionalLong.empty(), properties.getMaxLongPropertyValue());
        assertEquals(42, properties.getLong(0));
    }

    @Test
    void returnsValuesThatHaveBeenSet() {
        var properties = createNodeProperties(2L, 42.0, b -> b.set(1, Values.of(1.0)));

        assertEquals(1.0, properties.getDouble(1));
        assertEquals(1.0, properties.getDouble(1));
    }

    @Test
    void returnsDefaultOnMissingEntries() {
        var expectedImplicitDefault = 42.0;
        var properties = createNodeProperties(2L, expectedImplicitDefault, b -> {});

        assertEquals(expectedImplicitDefault, properties.getDouble(2));
    }

    @Test
    void returnNaNIfItWasSet() {
        var properties = createNodeProperties(2L, 42.0, b -> b.set(1, Values.of(Double.NaN)));

        assertEquals(42.0, properties.getDouble(0));
        assertEquals(Double.NaN, properties.getDouble(1));
    }

    @Test
    void trackMaxValue() {
        var properties = createNodeProperties(2L, 0.0, b -> {
            b.set(0, Values.of(42));
            b.set(1, Values.of(21));
        });
        var maxPropertyValue = properties.getMaxLongPropertyValue();
        assertTrue(maxPropertyValue.isPresent());
        assertEquals(42, maxPropertyValue.getAsLong());
    }

    @Test
    void hasSize() {
        var properties = createNodeProperties(2L, 0.0, b -> {
            b.set(0, Values.of(42.0));
            b.set(1, Values.of(21.0));
        });
        assertEquals(2, properties.size());
    }

    @Test
    void shouldHandleNullValues() {
        var builder = NodePropertiesFromStoreBuilder.of(
            100,
            AllocationTracker.EMPTY,
            DefaultValue.DEFAULT
        );

        builder.set(0, null);
        builder.set(1, Values.longValue(42L));

        var properties = builder.build();

        assertEquals(ValueType.LONG, properties.getType());
        assertEquals(DefaultValue.LONG_DEFAULT_FALLBACK, properties.getLong(0L));
        assertEquals(42L, properties.getLong(1L));
    }

    @Test
    void threadSafety() throws InterruptedException {
        var pool = Executors.newFixedThreadPool(2);
        var nodeSize = 100_000;
        var builder = NodePropertiesFromStoreBuilder.of(nodeSize, AllocationTracker.EMPTY, DefaultValue.of(Double.NaN));

        var phaser = new Phaser(3);
        pool.execute(() -> {
            // wait for start signal
            phaser.arriveAndAwaitAdvance();
            // first task, set the value 2 on every other node, except for 1338 which is set to 2^41
            // the idea is that the maxValue set will read the currentMax of 2, decide to update to 2^41 and write
            // that value, while the other thread will write 2^42 in the meantime. If that happens,
            // this thread would overwrite a new maxValue.
            for (int i = 0; i < nodeSize; i += 2) {
                builder.set(i, Values.of(i == 1338 ? 0x1p41 : 2.0));
            }
        });
        pool.execute(() -> {
            // wait for start signal
            phaser.arriveAndAwaitAdvance();
            // second task, sets the value 1 on every other node, except for 1337 which is set to 2^42
            // Depending on thread scheduling, the write for 2^42 might be overwritten by the first thread
            for (int i = 1; i < nodeSize; i += 2) {
                builder.set(i, Values.of(i == 1337 ? 0x1p42 : 1.0));
            }
        });

        phaser.arriveAndAwaitAdvance();

        pool.shutdown();
        pool.awaitTermination(10, TimeUnit.SECONDS);

        var properties = builder.build();
        for (int i = 0; i < nodeSize; i++) {
            var expected = i == 1338 ? 0x1p41 : i == 1337 ? 0x1p42 : i % 2 == 0 ? 2.0 : 1.0;
            assertEquals(expected, properties.getDouble(i), "" + i);
        }
        assertEquals(nodeSize, properties.size());
        var maxPropertyValue = properties.getMaxDoublePropertyValue();
        assertTrue(maxPropertyValue.isPresent());

        // If write were correctly ordered, this is always true
        // If, however, the write to maxValue were to be non-atomic
        // e.g. `this.maxValue = Math.max(value, this.maxValue);`
        // this would occasionally be 2^41.
        assertEquals(1L << 42, maxPropertyValue.getAsDouble());
    }

    static NodeProperties createNodeProperties(long size, Object defaultValue, Consumer<NodePropertiesFromStoreBuilder> buildBlock) {
        var builder = NodePropertiesFromStoreBuilder.of(size, AllocationTracker.EMPTY, DefaultValue.of(defaultValue));
        buildBlock.accept(builder);
        return builder.build();
    }
}
