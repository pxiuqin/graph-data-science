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
package org.neo4j.graphalgo.triangle;

import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicLongArray;
import org.neo4j.graphalgo.core.write.PropertyTranslator;
import org.neo4j.graphalgo.result.AbstractCommunityResultBuilder;
import org.neo4j.graphalgo.result.AbstractResultBuilder;
import org.neo4j.graphalgo.triangle.IntersectingTriangleCount.TriangleCountResult;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;

import java.util.Optional;

import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

final class TriangleCountCompanion {

    static final String DESCRIPTION =
        "Triangle counting is a community detection graph algorithm that is used to " +
        "determine the number of triangles passing through each node in the graph.";


    static PropertyTranslator<TriangleCountResult> nodePropertyTranslator() {
        return (PropertyTranslator.OfLong<TriangleCountResult>) (data, nodeId) -> data.localTriangles().get(nodeId);
    }

    static <PROC_RESULT, CONFIG extends TriangleCountBaseConfig> AbstractResultBuilder<PROC_RESULT> resultBuilder(
        TriangleCountResultBuilder<PROC_RESULT> procResultBuilder,
        AlgoBaseProc.ComputationResult<IntersectingTriangleCount, IntersectingTriangleCount.TriangleCountResult, CONFIG> computeResult
    ) {
        var result = Optional.ofNullable(computeResult.result())
            .orElse(EmptyResult.EMPTY_RESULT);

        return procResultBuilder
            .withTriangleCount(result.globalTriangles())
            .buildHistogram()
            .withCommunityFunction(result.localTriangles()::get);
    }

    abstract static class TriangleCountResultBuilder<PROC_RESULT> extends AbstractCommunityResultBuilder<PROC_RESULT> {

        long triangleCount = 0;

        TriangleCountResultBuilder(ProcedureCallContext callContext, AllocationTracker tracker) {
            super(callContext, tracker);
        }

        TriangleCountResultBuilder<PROC_RESULT> withTriangleCount(long triangleCount) {
            this.triangleCount = triangleCount;
            return this;
        }

        TriangleCountResultBuilder<PROC_RESULT> buildCommunityCount() {
            this.buildCommunityCount = true;
            return this;
        }

        TriangleCountResultBuilder<PROC_RESULT> buildHistogram() {
            this.buildHistogram = true;
            return this;
        }

    }

    private TriangleCountCompanion() {}

    private static final class EmptyResult implements IntersectingTriangleCount.TriangleCountResult {

        static final EmptyResult EMPTY_RESULT = new EmptyResult();

        private EmptyResult() {}

        @Override
        public HugeAtomicLongArray localTriangles() {
            return HugeAtomicLongArray.newArray(0, AllocationTracker.EMPTY);
        }

        @Override
        public long globalTriangles() {
            return 0;
        }

    }
}