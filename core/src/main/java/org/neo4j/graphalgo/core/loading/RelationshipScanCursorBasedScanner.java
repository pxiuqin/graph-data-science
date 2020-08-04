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
package org.neo4j.graphalgo.core.loading;

import org.neo4j.graphalgo.compat.Neo4jProxy;
import org.neo4j.graphalgo.core.SecureTransaction;
import org.neo4j.internal.kernel.api.RelationshipScanCursor;
import org.neo4j.internal.kernel.api.Scan;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.impl.store.NeoStores;
import org.neo4j.kernel.impl.store.RelationshipStore;

public final class RelationshipScanCursorBasedScanner extends AbstractCursorBasedScanner<RelationshipReference, RelationshipScanCursor, RelationshipStore, Void> {

    public static final StoreScanner.Factory<RelationshipReference> FACTORY = RelationshipScanCursorBasedScanner::new;

    private RelationshipScanCursorBasedScanner(int prefetchSize, SecureTransaction transaction) {
        super(prefetchSize, transaction, null);
    }

    @Override
    RelationshipStore store(NeoStores neoStores) {
        return neoStores.getRelationshipStore();
    }

    @Override
    RelationshipScanCursor entityCursor(KernelTransaction transaction) {
        return Neo4jProxy.allocateRelationshipScanCursor(transaction.cursors(), transaction.pageCursorTracer());
    }

    @Override
    Scan<RelationshipScanCursor> entityCursorScan(KernelTransaction transaction, Void ignore) {
        return transaction.dataRead().allRelationshipsScan();
    }

    @Override
    RelationshipReference cursorReference(KernelTransaction transaction, RelationshipScanCursor cursor) {
        return new RelationshipScanCursorReference(cursor);
    }
}
