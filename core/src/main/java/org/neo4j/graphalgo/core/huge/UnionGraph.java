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

import com.carrotsearch.hppc.BitSet;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.api.CSRGraph;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.ImmutableTopology;
import org.neo4j.graphalgo.api.NodeMapping;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.RelationshipConsumer;
import org.neo4j.graphalgo.api.RelationshipCursor;
import org.neo4j.graphalgo.api.RelationshipIntersect;
import org.neo4j.graphalgo.api.RelationshipWithPropertyConsumer;
import org.neo4j.graphalgo.api.Relationships;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterable;
import org.neo4j.graphalgo.core.utils.collection.primitive.PrimitiveLongIterator;

import java.util.Collection;
import java.util.Set;
import java.util.function.LongPredicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class UnionGraph implements CSRGraph {

    private final CSRGraph first;
    private final Collection<? extends CSRGraph> graphs;

    public static CSRGraph of(Collection<? extends CSRGraph> graphs) {
        if (graphs.isEmpty()) {
            throw new IllegalArgumentException("no graphs");
        }
        if (graphs.size() == 1) {
            return graphs.iterator().next();
        }
        return new UnionGraph(graphs);
    }

    private UnionGraph(Collection<? extends CSRGraph> graphs) {
        first = graphs.iterator().next();
        this.graphs = graphs;
    }

    @Override
    public long nodeCount() {
        return first.nodeCount();
    }

    @Override
    public NodeMapping nodeMapping() {
        return first.nodeMapping();
    }

    @Override
    public long relationshipCount() {
        return graphs.stream().mapToLong(Graph::relationshipCount).sum();
    }

    @Override
    public Collection<PrimitiveLongIterable> batchIterables(final int batchSize) {
        return first.batchIterables(batchSize);
    }

    @Override
    public void forEachNode(LongPredicate consumer) {
        first.forEachNode(consumer);
    }

    @Override
    public PrimitiveLongIterator nodeIterator() {
        return first.nodeIterator();
    }

    @Override
    public NodeProperties nodeProperties(final String propertyKey) {
        return first.nodeProperties(propertyKey);
    }

    @Override
    public Set<String> availableNodeProperties() {
        return first.availableNodeProperties();
    }

    @Override
    public long toMappedNodeId(long nodeId) {
        return first.toMappedNodeId(nodeId);
    }

    @Override
    public long toOriginalNodeId(long nodeId) {
        return first.toOriginalNodeId(nodeId);
    }

    @Override
    public boolean contains(final long nodeId) {
        return first.contains(nodeId);
    }

    @Override
    public double relationshipProperty(final long sourceNodeId, final long targetNodeId, double fallbackValue) {
        return first.relationshipProperty(sourceNodeId, targetNodeId, fallbackValue);
    }

    @Override
    public double relationshipProperty(long sourceNodeId, long targetNodeId) {
        return first.relationshipProperty(sourceNodeId, targetNodeId);
    }

    @Override
    public void forEachRelationship(long nodeId, RelationshipConsumer consumer) {
        for (Graph graph : graphs) {
            graph.forEachRelationship(nodeId, consumer);
        }
    }

    @Override
    public void forEachRelationship(long nodeId, double fallbackValue, RelationshipWithPropertyConsumer consumer) {
        for (Graph graph : graphs) {
            graph.forEachRelationship(nodeId, fallbackValue, consumer);
        }
    }

    @Override
    public Stream<RelationshipCursor> streamRelationships(long nodeId, double fallbackValue) {
        return graphs
            .stream()
            .flatMap(graph -> graph.streamRelationships(nodeId, fallbackValue));
    }

    @Override
    public int degree(long nodeId) {
        return Math.toIntExact(graphs.stream().mapToLong(g -> g.degree(nodeId)).sum());
    }

    @Override
    public int degreeWithoutParallelRelationships(long nodeId) {
        if (!isMultiGraph()) {
            return degree(nodeId);
        }
        var degreeCounter = new ParallelRelationshipDegreeCounter();
        graphs.forEach(graph -> graph.forEachRelationship(nodeId, degreeCounter));
        return degreeCounter.degree();
    }

    @Override
    public CSRGraph concurrentCopy() {
        return of(graphs.stream().map(CSRGraph::concurrentCopy).collect(Collectors.toList()));
    }

    @Override
    public RelationshipIntersect intersection(long maxDegree) {
        return new UnionGraphIntersect((CompositeAdjacencyList) relationshipTopology().list(), maxDegree);
    }

    /**
     * O(n) !
     */
    @Override
    public boolean exists(long sourceNodeId, long targetNodeId) {
        return graphs.stream().anyMatch(g -> g.exists(sourceNodeId, targetNodeId));
    }

    /*
     * O(n) !
     */
    @Override
    public long getTarget(long sourceNodeId, long index) {
        return graphs.stream()
                .mapToLong(g -> g.getTarget(sourceNodeId, index))
                .filter(t -> t != HugeGraph.GetTargetConsumer.TARGET_NOT_FOUND)
                .findFirst()
                .orElse(HugeGraph.GetTargetConsumer.TARGET_NOT_FOUND);
    }

    @Override
    public void canRelease(boolean canRelease) {
        for (Graph graph : graphs) {
            graph.canRelease(canRelease);
        }
    }

    @Override
    public void releaseTopology() {
        for (Graph graph : graphs) {
            graph.releaseTopology();
        }
    }

    @Override
    public void releaseProperties() {
        for (Graph graph : graphs) {
            graph.releaseProperties();
        }
    }

    @Override
    public boolean hasRelationshipProperty() {
        return first.hasRelationshipProperty();
    }

    @Override
    public boolean isUndirected() {
        return graphs.stream().allMatch(Graph::isUndirected);
    }

    @Override
    public boolean isMultiGraph() {
        // we need to run a check across all relationships between the sub-graphs of the union
        // maybe we'll do that later; for now union never guarantees parallel-free
        return true;
    }

    @Override
    public Relationships.Topology relationshipTopology() {
        var adjacencyLists = graphs
            .stream()
            .map(graph -> graph.relationshipTopology().list())
            .collect(Collectors.toList());
        var adjacencyOffsets = graphs
            .stream()
            .map(graph -> graph.relationshipTopology().offsets())
            .collect(Collectors.toList());

        return ImmutableTopology.builder()
            .offsets(new CompositeAdjacencyOffsets(adjacencyOffsets))
            .list(new CompositeAdjacencyList(adjacencyLists, adjacencyOffsets))
            .orientation(Orientation.NATURAL)
            .elementCount(relationshipCount())
            .isMultiGraph(true)
            .build();
    }

    private static class ParallelRelationshipDegreeCounter implements RelationshipConsumer {
        private final BitSet visited;

        ParallelRelationshipDegreeCounter() {
            visited = BitSet.newInstance();
        }

        @Override
        public boolean accept(long s, long t) {
            visited.set(t);
            return true;
        }

        int degree() {
            return Math.toIntExact(visited.cardinality());
        }
    }
}
