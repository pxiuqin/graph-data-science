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
package org.neo4j.gds.embeddings.graphsage.ddl4j.tensor;

import org.neo4j.gds.embeddings.graphsage.ddl4j.Dimensions;

public class Scalar extends Tensor<Scalar> {

    public Scalar(double value) {
        super(new double[] {value}, Dimensions.scalar());
    }

    @Override
    public Scalar zeros() {
        return new Scalar(0D);
    }

    @Override
    public Scalar copy() {
        return new Scalar(value());
    }

    @Override
    public Scalar add(Scalar b) {
        return new Scalar(value() + b.value());
    }

    private double value() {
        return data[0];
    }
}
