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
package org.neo4j.graphalgo.api;

import org.neo4j.graphalgo.NodeLabel;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterable;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterator;

import java.util.Collection;
import java.util.Set;
import java.util.function.LongPredicate;

public abstract class FilterGraph implements Graph {

    protected final Graph graph;

    public FilterGraph(Graph graph) {
        this.graph = graph;
    }

    @Override
    public long relationshipCount() {
        return graph.relationshipCount();  //关系数量
    }

    @Override
    public boolean isUndirected() {
        return graph.isUndirected();
    }

    @Override
    public boolean hasRelationshipProperty() {
        return graph.hasRelationshipProperty();
    }

    @Override
    public void canRelease(boolean canRelease) {
        graph.canRelease(canRelease);
    }

    @Override
    public RelationshipIntersect intersection(long maxDegree) {
        return graph.intersection(maxDegree);
    }

    @Override
    public Collection<PrimitiveLongIterable> batchIterables(int batchSize) {
        return graph.batchIterables(batchSize);
    }

    @Override
    public int degree(long nodeId) {
        return graph.degree(nodeId);
    }

    @Override
    public NodeMapping nodeMapping() {
        return graph.nodeMapping();
    }

    @Override
    public long toMappedNodeId(long nodeId) {
        return graph.toMappedNodeId(nodeId);
    }

    @Override
    public long toOriginalNodeId(long nodeId) {
        return graph.toOriginalNodeId(nodeId);
    }

    @Override
    public boolean contains(long nodeId) {
        return graph.contains(nodeId);
    }

    @Override
    public long nodeCount() {
        return graph.nodeCount();
    }

    @Override
    public void forEachNode(LongPredicate consumer) {
        graph.forEachNode(consumer);
    }

    @Override
    public PrimitiveLongIterator nodeIterator() {
        return graph.nodeIterator();
    }

    @Override
    public Set<NodeLabel> nodeLabels(long nodeId) {
        return graph.nodeLabels(nodeId);
    }

    @Override
    public Set<NodeLabel> availableNodeLabels() {
        return graph.availableNodeLabels();
    }

    @Override
    public NodeProperties nodeProperties(String propertyKey) {
        return graph.nodeProperties(propertyKey);
    }

    @Override
    public Set<String> availableNodeProperties() {
        return graph.availableNodeProperties();
    }

    @Override
    public long getTarget(long nodeId, long index) {
        return graph.getTarget(nodeId, index);
    }

    @Override
    public void forEachRelationship(long nodeId, RelationshipConsumer consumer) {
        graph.forEachRelationship(nodeId, consumer);
    }

    @Override
    public void forEachRelationship(long nodeId, double fallbackValue, RelationshipWithPropertyConsumer consumer) {
        graph.forEachRelationship(nodeId, fallbackValue, consumer);
    }

    @Override
    public boolean exists(long sourceNodeId, long targetNodeId) {
        return graph.exists(sourceNodeId, targetNodeId);
    }

    @Override
    public double relationshipProperty(long sourceNodeId, long targetNodeId, double fallbackValue) {
        return graph.relationshipProperty(sourceNodeId, targetNodeId, fallbackValue);
    }

    @Override
    public double relationshipProperty(long sourceNodeId, long targetNodeId) {
        return graph.relationshipProperty(sourceNodeId, targetNodeId);
    }
}
