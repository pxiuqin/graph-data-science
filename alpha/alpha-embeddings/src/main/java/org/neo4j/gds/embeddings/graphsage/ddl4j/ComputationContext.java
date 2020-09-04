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

import org.neo4j.gds.embeddings.graphsage.ddl4j.functions.PassthroughVariable;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.tensor.TensorFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class ComputationContext {
    private final Map<Variable<?>, Tensor<?>> data;
    private final Map<Variable<?>, Tensor<?>> gradients;

    public ComputationContext() {
        this.data = new ConcurrentHashMap<>();
        this.gradients = new ConcurrentHashMap<>();
    }

    public Tensor<?> forward(Variable<?> variable) {
        for (Variable<?> parent : variable.parents()) {
            if (!data.containsKey(parent)) {
                Tensor<?> parentData = forward(parent);
                data.put(parent, parentData);
            }
        }
        return data.computeIfAbsent(variable, ignore -> variable.apply(this));
    }

    public Tensor<?> data(Variable<?> variable) {
        return data.get(variable);
    }

    public Tensor<?> gradient(Variable<?> variable) {
        return gradients.get(variable);
    }

    public void backward(Variable<?> function) {
        if (function.dimensions().length != 1 || data(function).totalSize() != 1) {
            throw new IllegalArgumentException("Backward requires a variable with rank 1 and single dimension of size 1.");
        }
        gradients.clear();
        Queue<BackPropTask> executionQueue = new LinkedBlockingQueue<>();
        PassthroughVariable<?> dummy = new PassthroughVariable<>(function);
        executionQueue.add(new BackPropTask(function, dummy));
        Map<Variable<?>, AtomicInteger> upstreamCounters = new HashMap<>();
        initUpstream(dummy, upstreamCounters);
        backward(executionQueue, upstreamCounters);
    }

    private void backward(Queue<BackPropTask> executionQueue, Map<Variable<?>, AtomicInteger> upstreamCounters) {
        while (!executionQueue.isEmpty()) {
            BackPropTask task = executionQueue.poll();
            var variable = task.variable;
            var child = task.child;
            Tensor<?> gradient = child.gradient(variable, this);
            updateGradient(variable, gradient);

            upstreamCounters.get(variable).decrementAndGet();
            if (upstreamCounters.get(variable).get() == 0) {
                for (Variable<?> parent : variable.parents()) {
                    if (parent.requireGradient()) {
                        executionQueue.offer(new BackPropTask(parent, variable));
                    }
                }
            }
        }
    }

    private void initUpstream(Variable<?> function, Map<Variable<?>, AtomicInteger> upstreamCounters) {
        for (Variable<?> parent : function.parents()) {
            if (parent.requireGradient()) {
                boolean firstToSeeParent = !upstreamCounters.containsKey(parent);
                if (firstToSeeParent) {
                    initUpstream(parent, upstreamCounters);
                    upstreamCounters.put(parent, new AtomicInteger(0));
                }
                upstreamCounters.get(parent).incrementAndGet();
            }
        }
    }

    private void updateGradient(Variable<?> variable, Tensor<?> gradient) {
        gradients.putIfAbsent(variable, TensorFactory.constant(0D, variable.dimensions()));
        gradients.get(variable).addInPlace(gradient);
    }

    static class BackPropTask {
        Variable<?> variable;
        Variable<?> child;

        BackPropTask(Variable<?> variable, Variable<?> child) {
            this.variable = variable;
            this.child = child;
        }
    }

}
