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
package org.neo4j.graphalgo.utils;

import org.immutables.builder.Builder;

import java.util.Collections;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;

public final class AutoCloseableThreadLocal<T extends AutoCloseable> implements Supplier<T>, AutoCloseable {

    private final Consumer<? super T> destructor;
    private final CloseableThreadLocal<T> locals;
    private final Set<T> copies;

    public static <T extends AutoCloseable> AutoCloseableThreadLocal<T> withInitial(CheckedSupplier<T, ?> initial) {
        return new AutoCloseableThreadLocal<>(initial, Optional.empty());
    }

    @Builder.Constructor
    public AutoCloseableThreadLocal(
        @Builder.Parameter Supplier<T> constructor,
        Optional<Consumer<? super T>> destructor
    ) {
        this.destructor = destructor.orElse(doNothing -> {});
        this.copies = Collections.newSetFromMap(new ConcurrentHashMap<>());
        this.locals = CloseableThreadLocal.withInitial(() -> {
            var item = constructor.get();
            copies.add(item);
            return item;
        });
    }

    @Override
    public T get() {
        return locals.get();
    }

    @Override
    public void close() {
        var error = new AtomicReference<RuntimeException>();
        copies.removeIf(item -> {
            try (item) {
                destructor.accept(item);
            } catch (RuntimeException e) {
                error.set(ExceptionUtil.chain(error.get(), e));
            } catch (Exception e) {
                error.set(ExceptionUtil.chain(error.get(), new RuntimeException(e)));
            }
            return true;
        });
        locals.close();
        var errorWhileClosing = error.get();
        if (errorWhileClosing != null) {
            throw errorWhileClosing;
        }
    }
}
