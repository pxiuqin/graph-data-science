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
package org.neo4j.graphalgo.beta.pregel.bfs;

import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.beta.pregel.NodeSchemaBuilder;
import org.neo4j.graphalgo.beta.pregel.Pregel;
import org.neo4j.graphalgo.beta.pregel.PregelComputation;
import org.neo4j.graphalgo.beta.pregel.PregelContext;
import org.neo4j.graphalgo.beta.pregel.annotation.GDSMode;
import org.neo4j.graphalgo.beta.pregel.annotation.PregelProcedure;

/**
 * Setting the value for each node to the level/iteration the node is discovered via BFS.
 */
@PregelProcedure(name = "example.pregel.bfs", modes = {GDSMode.STREAM})
public class BFSLevelPregel implements PregelComputation<BFSPregelConfig> {

    private static final long NOT_FOUND = -1;
    public static final String LEVEL = "LEVEL";

    @Override
    public Pregel.NodeSchema nodeSchema() {
        return new NodeSchemaBuilder()
            .putElement(LEVEL, ValueType.LONG)
            .build();
    }

    @Override
    public void compute(PregelContext.ComputeContext<BFSPregelConfig> context, Pregel.Messages messages) {
        if (context.isInitialSuperstep()) {
            if (context.nodeId() == context.config().startNode()) {
                context.setNodeValue(LEVEL, 0);
                context.sendToNeighbors(1);
                context.voteToHalt();
            } else {
                context.setNodeValue(LEVEL, NOT_FOUND);
            }
        } else if (messages.iterator().hasNext()) {
            long level = context.longNodeValue(LEVEL);
            if (level == NOT_FOUND) {
                level = messages.iterator().next().longValue();

                context.setNodeValue(LEVEL, level);
                context.sendToNeighbors(level + 1);
            }
            context.voteToHalt();
        }
    }
}
