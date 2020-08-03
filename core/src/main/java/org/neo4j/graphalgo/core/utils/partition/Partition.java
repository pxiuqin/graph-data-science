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
package org.neo4j.graphalgo.core.utils.partition;

//分片处理，这里主要通过开始节点和节点数量来分区【按照节点来处理并行数据，通过给定开始节点和结束节点确定节点处理的范围】
public class Partition {

    public static final int MAX_NODE_COUNT = (Integer.MAX_VALUE - 32) >> 1;

    public final long startNode;
    public final long nodeCount;

    public Partition(long startNode, long nodeCount) {
        this.startNode = startNode;
        this.nodeCount = nodeCount;
    }

    public boolean fits(int otherPartitionsCount) {
        return MAX_NODE_COUNT - otherPartitionsCount >= nodeCount;
    }
}
