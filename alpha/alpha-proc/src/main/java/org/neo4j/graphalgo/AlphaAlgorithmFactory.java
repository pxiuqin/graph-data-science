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
package org.neo4j.graphalgo;

import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.core.utils.ProgressLoggerAdapter;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.exceptions.MemoryEstimationNotImplementedException;
import org.neo4j.logging.Log;

@FunctionalInterface
public interface AlphaAlgorithmFactory<ALGO extends Algorithm<ALGO, ?>, CONFIG extends AlgoBaseConfig> extends AlgorithmFactory<ALGO, CONFIG> {
    @Override
    default ALGO build(Graph graph, CONFIG configuration, AllocationTracker tracker, Log log) {
        ALGO algo = buildAlphaAlgo(graph, configuration, tracker, log);
        return algo.withProgressLogger(new ProgressLoggerAdapter(log, algo.getClass().getSimpleName()));
    }

    ALGO buildAlphaAlgo(Graph graph, CONFIG configuration, AllocationTracker tracker, Log log);

    @Override
    default MemoryEstimation memoryEstimation(CONFIG configuration) {
        throw new MemoryEstimationNotImplementedException();
    }
}
