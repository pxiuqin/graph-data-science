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

import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.values.storable.Value;
import org.neo4j.values.storable.Values;

/**
 * {@link NodeProperties} implementation which always returns
 * a given default property value upon invocation.
 */
public abstract class NullPropertyMap implements NodeProperties {

    static public class DoubleNullPropertyMap extends NullPropertyMap {
        private final double defaultValue;

        public DoubleNullPropertyMap(double defaultValue) {this.defaultValue = defaultValue;}

        @Override
        public double getDouble(long nodeId) {
            return this.defaultValue;
        }

        @Override
        public double getDouble(long nodeId, double defaultValue) {
            return defaultValue;
        }

        @Override
        public Value getValue(long nodeId) {
            return Values.doubleValue(defaultValue);
        }

        @Override
        public ValueType getType() {
            return ValueType.DOUBLE;
        }
    }

    static public class LongNullPropertyMap extends NullPropertyMap {
        private final long defaultValue;

        public LongNullPropertyMap(long defaultValue) {this.defaultValue = defaultValue;}

        @Override
        public long getLong(long nodeId) {
            return this.defaultValue;
        }

        @Override
        public long getLong(long nodeId, long defaultValue) {
            return defaultValue;
        }

        @Override
        public Value getValue(long nodeId) {
            return Values.longValue(defaultValue);
        }

        @Override
        public ValueType getType() {
            return ValueType.LONG;
        }
    }

}
