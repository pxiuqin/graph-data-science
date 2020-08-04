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

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.mult.MatrixMatrixMult_DDRM;
import org.neo4j.gds.embeddings.graphsage.ddl4j.ComputationContext;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Dimensions;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Variable;

import java.util.List;

public class MatrixMultiplyWithTransposedSecondOperand extends Variable {

    private final Variable A;
    private final Variable B;

    public MatrixMultiplyWithTransposedSecondOperand(Variable A, Variable B) {
        // the dimensions of a matrix multiplication of dimensions (m, n) x (o, p) = (m, p)
        // B is being transposed as Bt and its dimensions are (p, o) so the result will be dimensions (m, o)
        super(List.of(A, B), Dimensions.matrix(A.dimension(0), B.dimension(0)));
        this.A = A;
        this.B = B;
    }

    @Override
    protected Tensor apply(ComputationContext ctx) {
        Tensor t1 = ctx.data(A);
        Tensor t2 = ctx.data(B);
        return multiplyTransB(t1, t2);
    }

    @Override
    protected Tensor gradient(Variable parent, ComputationContext ctx) {
        Tensor gradient = ctx.gradient(this);
        if (parent == A) {
            return multiply(gradient, ctx.data(B));
        } else {
            return multiplyTransA(gradient, ctx.data(A));
        }
    }

    private Tensor multiply(Tensor t1, Tensor t2) {
        DMatrixRMaj m1 = DMatrixRMaj.wrap(t1.dimensions[0], t1.dimensions[1], t1.data);
        DMatrixRMaj m2 = DMatrixRMaj.wrap(t2.dimensions[0], t2.dimensions[1], t2.data);
        DMatrixRMaj prod = new DMatrixRMaj(m1.numRows, m2.numCols);
        MatrixMatrixMult_DDRM.mult_reorder(m1, m2, prod);
        return Tensor.matrix(prod.getData(), prod.numRows, prod.numCols);
    }

    private Tensor multiplyTransB(Tensor t1, Tensor t2) {
        DMatrixRMaj m1 = DMatrixRMaj.wrap(t1.dimensions[0], t1.dimensions[1], t1.data);
        DMatrixRMaj m2 = DMatrixRMaj.wrap(t2.dimensions[0], t2.dimensions[1], t2.data);
        DMatrixRMaj prod = new DMatrixRMaj(m1.numRows, m2.numRows);
        MatrixMatrixMult_DDRM.multTransB(m1, m2, prod);
        return Tensor.matrix(prod.getData(), prod.numRows, prod.numCols);
    }

    private Tensor multiplyTransA(Tensor t1, Tensor t2) {
        DMatrixRMaj m1 = DMatrixRMaj.wrap(t1.dimensions[0], t1.dimensions[1], t1.data);
        DMatrixRMaj m2 = DMatrixRMaj.wrap(t2.dimensions[0], t2.dimensions[1], t2.data);
        DMatrixRMaj prod = new DMatrixRMaj(m1.numCols, m2.numCols);
        MatrixMatrixMult_DDRM.multTransA_reorder(m1, m2, prod);
        return Tensor.matrix(prod.getData(), prod.numRows, prod.numCols);
    }

    public static MatrixMultiplyWithTransposedSecondOperand of(Variable A, Variable B) {
        return new MatrixMultiplyWithTransposedSecondOperand(A, B);
    }
}
