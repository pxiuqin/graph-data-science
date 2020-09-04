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

/**
 * Bi-directional mapping between two id spaces.
 */
public interface IdMapping {

    /**
     * Defines the lower bound of mapped ids
     * TODO: function?
     */
    long START_NODE_ID = 0;

    /**
     * Map original nodeId to inner nodeId
     */
    long toMappedNodeId(long nodeId);

    /**
     * Map inner nodeId back to original nodeId
     */
    long toOriginalNodeId(long nodeId);

    /**
     * Returns true iff the nodeId is mapped, otherwise false.
     */
    boolean contains(long nodeId);

    /**
     * Number of mapped nodeIds.
     */
    long nodeCount();

    default IdMapping cloneIdMapping() {
        return this;
    }
}
