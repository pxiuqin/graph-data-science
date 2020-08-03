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
package org.neo4j.graphalgo.pagerank;

import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;

import java.util.concurrent.ExecutorService;

//带权重的度计算处理【利用了带权重的度中心性计算结果】
public class WeightedDegreeComputer implements DegreeComputer {

    private final Graph graph;
    private final boolean cacheWeights;

    WeightedDegreeComputer(Graph graph, boolean cacheWeights) {
        this.graph = graph;
        this.cacheWeights = cacheWeights;
    }

    @Override
    public DegreeCache degree(
            ExecutorService executor,
            int concurrency,
            AllocationTracker tracker) {
        WeightedDegreeCentrality degreeCentrality = new WeightedDegreeCentrality(
            graph,
            concurrency,
            cacheWeights,
            executor,
            tracker
        );
        degreeCentrality.compute();
        return new DegreeCache(degreeCentrality.degrees(), degreeCentrality.weights(), -1D);
    }
}
