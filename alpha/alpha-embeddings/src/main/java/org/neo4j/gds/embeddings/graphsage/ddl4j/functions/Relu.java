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

public class Relu extends SingleParentVariable {

    private static final double ALPHA = 0.01;

    public Relu(Variable parent) {
        super(parent, parent.dimensions());
    }

    @Override
    protected Tensor apply(ComputationContext ctx) {
        return ctx.data(parent).map(value -> value > 0 ? value : ALPHA * value);
    }

    @Override
    protected Tensor gradient(ComputationContext ctx) {
        return ctx.data(parent).map(value -> value > 0 ? 1 : ALPHA);
    }
}
