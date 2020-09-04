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
package org.neo4j.graphalgo.core.loading.nodeproperties;

import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.api.nodeproperties.FloatArrayNodeProperties;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeObjectArray;
import org.neo4j.graphalgo.utils.ValueConversion;
import org.neo4j.values.storable.Value;

public class FloatArrayNodePropertiesBuilder extends InnerNodePropertiesBuilder {

    private final HugeObjectArray<float[]> objectArray;
    private final DefaultValue defaultValue;

    public FloatArrayNodePropertiesBuilder(long nodeCount, DefaultValue defaultValue, AllocationTracker tracker) {
        this.defaultValue = defaultValue;
        this.objectArray = HugeObjectArray.newArray(float[].class, nodeCount, tracker);
    }

    @Override
    void setValue(long nodeId, Value value) {
        objectArray.set(nodeId, ValueConversion.getFloatArray(value));
    }

    @Override
    FloatArrayNodeProperties build(long size) {
        return new FloatArrayStoreNodeProperties(objectArray, defaultValue, size);
    }

    static class FloatArrayStoreNodeProperties implements FloatArrayNodeProperties {
        private final HugeObjectArray<float[]> propertyValues;
        private final DefaultValue defaultValue;
        private final long size;

        FloatArrayStoreNodeProperties(
            HugeObjectArray<float[]> propertyValues,
            DefaultValue defaultValue,
            long size
        ) {
            this.propertyValues = propertyValues;
            this.defaultValue = defaultValue;
            this.size = size;
        }

        @Override
        public float[] floatArrayValue(long nodeId) {
            float[] data = propertyValues.get(nodeId);
            if (data == null) {
                return defaultValue.floatArrayValue();
            }
            return data;
        }

        @Override
        public long size() {
            return size;
        }
    }
}
