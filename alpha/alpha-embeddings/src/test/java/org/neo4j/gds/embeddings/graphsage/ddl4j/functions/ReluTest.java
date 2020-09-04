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
package org.neo4j.gds.embeddings.graphsage.ddl4j.functions;

import org.junit.jupiter.api.Test;
import org.neo4j.gds.embeddings.graphsage.ddl4j.GraphSageBaseTest;
import org.neo4j.gds.embeddings.graphsage.ddl4j.FiniteDifferenceTest;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Variable;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.helper.Constant;
import org.neo4j.gds.embeddings.graphsage.ddl4j.helper.ElementSum;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Vector;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ReluTest extends GraphSageBaseTest implements FiniteDifferenceTest {
    @Override
    public double epsilon() {
        return 1E-8;
    }

    @Test
    void shouldApproximateGradientScalarValued() {
        double[] vectorData = {0.1};
        Weights<Vector> weights = new Weights<>(new Vector(vectorData));

        finiteDifferenceShouldApproximateGradient(weights, new Relu<>(weights));
    }

    @Test
    void shouldApproximateGradient() {
        double[] vectorData = {-1, 5, 2};
        Weights<Vector> weights = new Weights<>(new Vector(vectorData));

        finiteDifferenceShouldApproximateGradient(weights, new ElementSum(List.of(new Relu<>(weights))));
    }

    @Test
    void shouldComputeRelu() {
        double[] vectorData = {14, -5, 36, 0};
        Constant<Vector> p = Constant.vector(vectorData);

        Variable<Vector> relu = new Relu<>(p);

        Tensor<?> resultData = ctx.forward(relu);
        assertNotNull(resultData);
        assertEquals(vectorData.length, resultData.data().length);

        assertArrayEquals(new double[]{14, 0.01 * -5, 36, 0}, resultData.data());
    }

    @Test
    void returnsEmptyDataForEmptyVariable() {
        double[] vectorData = {};
        Constant<Vector> p = Constant.vector(vectorData);

        Variable<Vector> relu = new Relu<>(p);

        Tensor<?> resultData = ctx.forward(relu);
        assertNotNull(resultData);
        assertEquals(vectorData.length, resultData.data().length);

        assertArrayEquals(new double[]{}, resultData.data());

    }

}
