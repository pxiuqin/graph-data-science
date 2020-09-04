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
package org.neo4j.graphalgo.beta.pregel.lp;

import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.beta.pregel.NodeSchemaBuilder;
import org.neo4j.graphalgo.beta.pregel.Pregel;
import org.neo4j.graphalgo.beta.pregel.PregelComputation;
import org.neo4j.graphalgo.beta.pregel.PregelConfig;
import org.neo4j.graphalgo.beta.pregel.PregelContext;
import org.neo4j.graphalgo.beta.pregel.annotation.GDSMode;
import org.neo4j.graphalgo.beta.pregel.annotation.PregelProcedure;

import java.util.Arrays;

/**
 * Basic implementation potentially suffering from oscillating vertex states due to synchronous computation.
 */
@PregelProcedure(name = "example.pregel.lp", modes = {GDSMode.STREAM})
public class LabelPropagationPregel implements PregelComputation<PregelConfig> {

    public static final String LABEL_KEY = "label";

    @Override
    public Pregel.NodeSchema nodeSchema() {
        return new NodeSchemaBuilder().putElement(LABEL_KEY, ValueType.LONG).build();
    }

    @Override
    public void init(PregelContext.InitContext<PregelConfig> context) {
        context.setNodeValue(LABEL_KEY, context.nodeId());
    }

    @Override
    public void compute(PregelContext.ComputeContext<PregelConfig> context, Pregel.Messages messages) {
        if (context.isInitialSuperstep()) {
            context.sendToNeighbors(context.nodeId());
        } else {
            if (messages != null) {
                long oldValue = context.longNodeValue(LABEL_KEY);
                long newValue = oldValue;

                // TODO: could be shared across compute functions per thread
                // We receive at most |degree| messages
                long[] buffer = new long[context.degree()];

                int messageCount = 0;

                for (var message : messages) {
                    buffer[messageCount++] = message.longValue();
                }

                int maxOccurences = 1;
                if (messageCount > 1) {
                    // Sort to compute the most frequent id
                    Arrays.sort(buffer, 0, messageCount);
                    int currentOccurences = 1;
                    for (int i = 1; i < messageCount; i++) {
                        if (buffer[i] == buffer[i - 1]) {
                            currentOccurences++;
                            if (currentOccurences > maxOccurences) {
                                maxOccurences = currentOccurences;
                                newValue = buffer[i];
                            }
                        } else {
                            currentOccurences = 1;
                        }
                    }
                }

                // All with same frequency, pick smallest id
                if (maxOccurences == 1) {
                    newValue = Math.min(oldValue, buffer[0]);
                }

                if (newValue != oldValue) {
                    context.setNodeValue(LABEL_KEY, newValue);
                    context.sendToNeighbors(newValue);
                }
            }
        }
        context.voteToHalt();
    }
}
