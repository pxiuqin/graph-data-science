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
package org.neo4j.graphalgo.api.nodeproperties;

import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.values.storable.Value;
import org.neo4j.values.storable.Values;

public interface DoubleArrayNodeProperties extends NodeProperties {

    @Override
    double[] doubleArrayValue(long nodeId);

    @Override
    default float[] floatArrayValue(long nodeId) {
        double[] doubleArray = doubleArrayValue(nodeId);

        if (doubleArray == null) {
            return null;
        } else {
            float[] floatArray = new float[doubleArray.length];
            for (int i = 0; i < doubleArray.length; i++) {
                floatArray[i] = (float) doubleArray[i];
            }
            return floatArray;
        }
    }

    @Override
    default Object getObject(long nodeId) {
        return doubleArrayValue(nodeId);
    }

    @Override
    default Value value(long nodeId) {
        var value = doubleArrayValue(nodeId);
        return value == null ? null : Values.doubleArray(value);
    };

    @Override
    default ValueType valueType() {
        return ValueType.DOUBLE_ARRAY;
    };

    @Override
    default double doubleValue(long nodeId) {
        throw new UnsupportedOperationException("double is not supported");
    };
}
