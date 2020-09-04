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
package org.neo4j.graphalgo.impl.shortestpaths;

import com.carrotsearch.hppc.BitSet;
import com.carrotsearch.hppc.IntArrayDeque;
import com.carrotsearch.hppc.IntDoubleMap;
import com.carrotsearch.hppc.IntDoubleScatterMap;
import com.carrotsearch.hppc.IntIntMap;
import com.carrotsearch.hppc.IntIntScatterMap;
import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.core.utils.ProgressLogger;
import org.neo4j.graphalgo.core.utils.queue.IntPriorityQueue;
import org.neo4j.graphalgo.core.utils.queue.SharedIntPriorityQueue;

import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static org.neo4j.graphalgo.core.heavyweight.Converters.longToIntConsumer;

public class ShortestPathAStar extends Algorithm<ShortestPathAStar, ShortestPathAStar> {

    private static final int PATH_END = -1;

    private Graph graph;
    private final int nodeCount;
    private final long startNode;
    private final long goalNode;
    private final NodeProperties latitudes;
    private final NodeProperties longitudes;
    private IntDoubleMap gCosts;
    private IntDoubleMap fCosts;
    private double totalCost;
    private IntPriorityQueue openNodes;
    private IntIntMap path;
    private IntArrayDeque shortestPath;
    private BitSet closedNodes;
    private final ProgressLogger progressLogger;

    public static final double NO_PATH_FOUND = -1.0;

    public ShortestPathAStar(
        Graph graph,
        long startNode,
        long goalNode,
        NodeProperties latitudes,
        NodeProperties longitudes
    ) {
        this.graph = graph;
        this.nodeCount = Math.toIntExact(graph.nodeCount());
        this.startNode = startNode;
        this.goalNode = goalNode;
        this.latitudes = latitudes;
        this.longitudes = longitudes;
        this.gCosts = new IntDoubleScatterMap(nodeCount);
        this.fCosts = new IntDoubleScatterMap(nodeCount);
        this.openNodes = SharedIntPriorityQueue.min(nodeCount, fCosts, Double.MAX_VALUE);
        this.path = new IntIntScatterMap(nodeCount);
        this.closedNodes = new BitSet(nodeCount);
        this.shortestPath = new IntArrayDeque();
        this.progressLogger = getProgressLogger();
    }

    @Override
    public ShortestPathAStar compute() {
        reset();
        int startNodeInternal = Math.toIntExact(graph.toMappedNodeId(startNode));
        double startNodeLat = getLatitude(startNodeInternal);
        double startNodeLon = getLongitude(startNodeInternal);
        int goalNodeInternal = Math.toIntExact(graph.toMappedNodeId(goalNode));
        double goalNodeLat = getLatitude(goalNodeInternal);
        double goalNodeLon = getLongitude(goalNodeInternal);
        double initialHeuristic = computeHeuristic(startNodeLat, startNodeLon, goalNodeLat, goalNodeLon);
        gCosts.put(startNodeInternal, 0.0);
        fCosts.put(startNodeInternal, initialHeuristic);
        openNodes.add(startNodeInternal, 0.0);
        run(goalNodeInternal);
        if (path.containsKey(goalNodeInternal)) {
            totalCost = gCosts.get(goalNodeInternal);
            int node = goalNodeInternal;
            while (node != PATH_END) {
                shortestPath.addFirst(node);
                node = path.getOrDefault(node, PATH_END);
            }
        }
        return this;
    }

    private void run(int goalNodeId) {
        double goalLat = getLatitude(goalNodeId);
        double goalLon = getLongitude(goalNodeId);
        while (!openNodes.isEmpty() && running()) {
            int currentNodeId = openNodes.pop();
            if (currentNodeId == goalNodeId) {
                return;
            }
            closedNodes.set(currentNodeId);
            double currentNodeCost = this.gCosts.getOrDefault(currentNodeId, Double.MAX_VALUE);
            graph.forEachRelationship(
                    currentNodeId,
                    1.0D,
                    longToIntConsumer((source, target, weight) -> {
                        double neighbourLat = getLatitude(target);
                        double neighbourLon = getLongitude(target);
                        double heuristic = computeHeuristic(neighbourLat, neighbourLon, goalLat, goalLon);
                        boolean weightChanged = updateCosts(source, target, weight + currentNodeCost, heuristic);
                        if (!closedNodes.get(target)) {
                            if (weightChanged) {
                                openNodes.update(target);
                            } else {
                                openNodes.add(target, 0);
                            }
                        }
                        return true;
                    }));
            progressLogger.logProgress((double) currentNodeId / (nodeCount - 1));
        }
    }

    private double computeHeuristic(double lat1, double lon1, double lat2, double lon2) {
        final int earthRadius = 6371;
        final double kmToNM = 0.539957;
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double distance = earthRadius * c * kmToNM;
        return distance;
    }

    private double getLongitude(long nodeId) {
        return longitudes.doubleValue(nodeId);
    }

    private double getLatitude(long nodeId) {
        return latitudes.doubleValue(nodeId);
    }

    private boolean updateCosts(int source, int target, double newCost, double heuristic) {
        double oldCost = gCosts.getOrDefault(target, Double.MAX_VALUE);
        if (newCost < oldCost) {
            gCosts.put(target, newCost);
            fCosts.put(target, newCost + heuristic);
            path.put(target, source);
            return oldCost < Double.MAX_VALUE;
        }
        return false;
    }

    private void reset() {
        closedNodes.clear();
        openNodes.clear();
        gCosts.clear();
        fCosts.clear();
        path.clear();
        shortestPath.clear();
        totalCost = NO_PATH_FOUND;
    }

    public Stream<Result> resultStream() {
        return StreamSupport.stream(shortestPath.spliterator(), false)
                .map(cursor -> new Result(graph.toOriginalNodeId(cursor.value), gCosts.get(cursor.value)));
    }

    public IntArrayDeque getFinalPath() {
        return shortestPath;
    }

    public double getTotalCost() {
        return totalCost;
    }

    public int getPathLength() {
        return shortestPath.size();
    }

    @Override
    public ShortestPathAStar me() {
        return this;
    }

    @Override
    public void release() {
        graph = null;
        gCosts = null;
        fCosts = null;
        openNodes = null;
        path = null;
        shortestPath = null;
        closedNodes = null;
    }

    public static class Result {

        /**
         * the neo4j node id
         */
        public final Long nodeId;
        /**
         * cost to reach the node from startNode
         */
        public final Double cost;

        public Result(Long nodeId, Double cost) {
            this.nodeId = nodeId;
            this.cost = cost;
        }
    }
}
