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
package org.neo4j.graphalgo.nodesim;

import org.neo4j.graphalgo.api.Graph;

//相似图结果
public class SimilarityGraphResult {
    private final Graph similarityGraph;
    private final long comparedNodes;
    private final boolean isTopKGraph;

    SimilarityGraphResult(Graph similarityGraph, long comparedNodes, boolean isTopKGraph) {
        this.similarityGraph = similarityGraph;
        this.comparedNodes = comparedNodes;
        this.isTopKGraph = isTopKGraph;
    }

    public Graph similarityGraph() {
        return similarityGraph;
    }

    public long comparedNodes() {
        return comparedNodes;
    }

    public boolean isTopKGraph() {
        return isTopKGraph;
    }
}
