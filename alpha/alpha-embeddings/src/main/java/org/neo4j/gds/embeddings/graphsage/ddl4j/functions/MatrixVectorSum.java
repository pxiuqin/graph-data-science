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

import org.neo4j.gds.embeddings.graphsage.ddl4j.AbstractVariable;
import org.neo4j.gds.embeddings.graphsage.ddl4j.ComputationContext;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Variable;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Matrix;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Vector;

import java.util.List;

import static org.neo4j.gds.embeddings.graphsage.ddl4j.Dimensions.COLUMNS_INDEX;
import static org.neo4j.gds.embeddings.graphsage.ddl4j.Dimensions.ROWS_INDEX;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public class MatrixVectorSum extends AbstractVariable<Matrix> {

    private final Variable<Matrix> matrix;
    private final Variable<Vector> vector;
    private final int rows;
    private final int cols;

    public MatrixVectorSum(Variable<Matrix> matrix, Variable<Vector> vector) {
        super(List.of(matrix, vector), matrix.dimensions());
        assert matrix.dimension(COLUMNS_INDEX) == vector.dimension(ROWS_INDEX) : formatWithLocale(
            "Cannot broadcast vector with length %d to a matrix with %d columns",
            vector.dimension(ROWS_INDEX),
            matrix.dimension(COLUMNS_INDEX)
        );
        this.matrix = matrix;
        this.rows = matrix.dimension(ROWS_INDEX);
        this.cols = matrix.dimension(COLUMNS_INDEX);
        this.vector = vector;
    }

    @Override
    public Matrix apply(ComputationContext ctx) {

        double[] matrixData = ctx.data(matrix).data();
        double[] vectorData = ctx.data(vector).data();

        double[] result = new double[matrixData.length];

        for(int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int matrixIndex = row * cols + col;
                result[matrixIndex] = matrixData[matrixIndex] + vectorData[col];
            }
        }

        return new Matrix(result, rows, cols);
    }

    @Override
    public Tensor<?> gradient(Variable<?> parent, ComputationContext ctx) {
        if (parent == matrix) {
            return ctx.gradient(this);
        } else {
            Tensor<?> gradient = ctx.gradient(this);
            double[] result = new double[cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    int matrixIndex = row * cols + col;
                    result[col] += gradient.dataAt(matrixIndex);
                }
            }

            return new Vector(result);
        }
    }
}
