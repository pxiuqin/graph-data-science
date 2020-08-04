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

import org.neo4j.gds.embeddings.graphsage.ddl4j.ComputationContext;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Variable;

import java.util.List;

public class ConstantScale extends Variable {
    private final Variable parent;
    private final double constant;

    public ConstantScale(Variable parent, double constant) {
        super(List.of(parent), parent.dimensions());
        this.parent = parent;
        this.constant = constant;
    }

    @Override
    protected Tensor apply(ComputationContext ctx) {
        return ctx.data(parent).scalarMultiply(constant);
    }

    @Override
    protected Tensor gradient(Variable parent, ComputationContext ctx) {
        return ctx.gradient(this).scalarMultiply(constant);
    }
}
