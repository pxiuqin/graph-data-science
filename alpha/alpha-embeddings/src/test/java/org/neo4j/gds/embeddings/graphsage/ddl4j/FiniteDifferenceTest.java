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
package org.neo4j.gds.embeddings.graphsage.ddl4j;

import org.neo4j.gds.embeddings.graphsage.ddl4j.functions.Weights;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public interface FiniteDifferenceTest {
    default double tolerance() {
        return 1E-5;
    }

    default double epsilon() {
        return 1E-4;
    }

    default void finiteDifferenceShouldApproximateGradient(Weights weightVariable, Variable loss) {
        finiteDifferenceShouldApproximateGradient(List.of(weightVariable), loss);
    }

    default void finiteDifferenceShouldApproximateGradient(List<Weights> weightVariables, Variable loss) {
        for (Weights variable : weightVariables) {
            for (int i = 0; i < Tensor.totalSize(variable.dimensions()); i++) {
                ComputationContext ctx = new ComputationContext();
                double f0 = ctx.forward(loss).data[0];
                ctx.backward(loss);
                var partialDerivative = ctx.gradient(variable).data[i];
                perturb(variable, i, epsilon());
                ComputationContext ctx2 = new ComputationContext();
                double f1 = ctx2.forward(loss).data[0];
                assertEquals(
                    (f1 - f0) / epsilon(),
                    partialDerivative,
                    tolerance(),
                    formatWithLocale("AutoGrad and FiniteDifference gradients differ for coordinate %s more than the tolerance", i)
                );
            }
        }
    }

    private void perturb(Weights variable, int index, double epsilon) {
        variable.data().data[index] += epsilon;
    }

}
