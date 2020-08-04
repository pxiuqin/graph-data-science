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
package org.neo4j.graphalgo.core.huge;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.values.storable.Value;

import java.util.OptionalLong;

public class FilteredNodeProperties implements NodeProperties {
    protected final NodeProperties properties;
    protected NodeFilteredGraph graph;

    public FilteredNodeProperties(NodeProperties properties, NodeFilteredGraph graph) {
        this.properties = properties;
        this.graph = graph;
    }

    @Override
    public double getDouble(long nodeId) {
        return properties.getDouble(translateId(nodeId));
    }

    @Override
    public double getDouble(long nodeId, double defaultValue) {
        return properties.getDouble(translateId(nodeId), defaultValue);
    }

    @Override
    public long getLong(long nodeId) {
        return properties.getLong(translateId(nodeId));
    }

    @Override
    public long getLong(long nodeId, long defaultValue) {
        return properties.getLong(translateId(nodeId), defaultValue);
    }

    @Override
    public double[] getDoubleArray(long nodeId) {
        return properties.getDoubleArray(translateId(nodeId));
    }

    @Override
    public double[] getDoubleArray(long nodeId, double[] defaultValue) {
        return properties.getDoubleArray(translateId(nodeId), defaultValue);
    }

    @Override
    public Object getObject(long nodeId) {
        return properties.getObject(translateId(nodeId));
    }

    @Override
    public Object getObject(long nodeId, Object defaultValue) {
        return properties.getObject(translateId(nodeId), defaultValue);
    }

    @Override
    public Value getValue(long nodeId) {
        return properties.getValue(translateId(nodeId));
    }

    @Override
    public ValueType getType() {
        return properties.getType();
    }

    @Override
    public OptionalLong getMaxPropertyValue() {
        MutableDouble currentMax = new MutableDouble(Double.NEGATIVE_INFINITY);
        graph.forEachNode(id -> {
            currentMax.setValue(Math.max(currentMax.doubleValue(), nodeProperty(id, Double.MIN_VALUE)));
            return true;
        });
        return currentMax.doubleValue() == Double.NEGATIVE_INFINITY
            ? OptionalLong.empty()
            : OptionalLong.of((long) currentMax.doubleValue());
    }

    @Override
    public long release() {
        long releasedFromProps = properties.release();
        graph = null;
        return releasedFromProps;
    }

    @Override
    public long size() {
        return Math.min(properties.size(), graph.nodeCount());
    }

    protected long translateId(long nodeId) {
        return graph.getIntermediateOriginalNodeId(nodeId);
    }
}
