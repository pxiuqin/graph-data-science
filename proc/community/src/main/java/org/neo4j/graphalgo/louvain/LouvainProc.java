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
package org.neo4j.graphalgo.louvain;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.CommunityProcCompanion;
import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.api.nodeproperties.LongArrayNodeProperties;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.result.AbstractCommunityResultBuilder;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;

final class LouvainProc {

    static final String LOUVAIN_DESCRIPTION =
        "The Louvain method for community detection is an algorithm for detecting communities in networks.";

    private LouvainProc() {}

    static <CONFIG extends LouvainBaseConfig> NodeProperties nodeProperties(
        AlgoBaseProc.ComputationResult<Louvain, Louvain, CONFIG> computationResult,
        String resultProperty
    ) {
        var config = computationResult.config();
        var includeIntermediateCommunities = config.includeIntermediateCommunities();
        if (!includeIntermediateCommunities) {
            return CommunityProcCompanion.nodeProperties(
                computationResult,
                resultProperty,
                computationResult.result().finalDendrogram().asNodeProperties()
            );
        } else {
            return (LongArrayNodeProperties) computationResult.result()::getCommunities;
        }
    }

    static <PROC_RESULT, CONFIG extends LouvainBaseConfig> AbstractResultBuilder<PROC_RESULT> resultBuilder(
        LouvainResultBuilder<PROC_RESULT> procResultBuilder,
        AlgoBaseProc.ComputationResult<Louvain, Louvain, CONFIG> computeResult
    ) {
        Louvain result = computeResult.result();
        boolean nonEmpty = !computeResult.isGraphEmpty();

        return procResultBuilder
            .withLevels(nonEmpty ? result.levels() : 0)
            .withModularity(nonEmpty ? result.modularities()[result.levels() - 1] : 0)
            .withModularities(nonEmpty ? result.modularities() : new double[0])
            .withCommunityFunction(nonEmpty ? result::getCommunity : null);
    }

    abstract static class LouvainResultBuilder<PROC_RESULT> extends AbstractCommunityResultBuilder<PROC_RESULT> {

        long levels = -1;
        double[] modularities = new double[]{};
        double modularity = -1;

        LouvainResultBuilder(
            ProcedureCallContext context,
            AllocationTracker tracker
        ) {
            super(
                context,
                tracker
            );
        }

        LouvainResultBuilder<PROC_RESULT> withLevels(long levels) {
            this.levels = levels;
            return this;
        }

        LouvainResultBuilder<PROC_RESULT> withModularities(double[] modularities) {
            this.modularities = modularities;
            return this;
        }

        LouvainResultBuilder<PROC_RESULT> withModularity(double modularity) {
            this.modularity = modularity;
            return this;
        }
    }
}
