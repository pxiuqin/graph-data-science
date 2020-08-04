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

import org.neo4j.graphalgo.compat.CompositeNodeCursor;
import org.neo4j.graphalgo.compat.Neo4jProxy;
import org.neo4j.graphalgo.core.SecureTransaction;
import org.neo4j.internal.kernel.api.NodeLabelIndexCursor;
import org.neo4j.internal.kernel.api.Scan;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.impl.store.NeoStores;
import org.neo4j.kernel.impl.store.NodeStore;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

final class MultipleNodeLabelIndexBasedScanner extends AbstractCursorBasedScanner<NodeReference, CompositeNodeCursor, NodeStore, int[]> {

    private final int[] labelIds;

    MultipleNodeLabelIndexBasedScanner(int[] labelIds, int prefetchSize, SecureTransaction transaction) {
        super(prefetchSize, transaction, labelIds);
        this.labelIds = labelIds;
    }

    @Override
    NodeStore store(NeoStores neoStores) {
        return neoStores.getNodeStore();
    }

    @Override
    CompositeNodeCursor entityCursor(KernelTransaction transaction) {
        List<NodeLabelIndexCursor> cursors = Arrays
            .stream(labelIds)
            .mapToObj(i -> Neo4jProxy.allocateNodeLabelIndexCursor(transaction.cursors(), transaction.pageCursorTracer()))
            .collect(Collectors.toList());
        return Neo4jProxy.compositeNodeCursor(cursors, labelIds);
    }

    @Override
    Scan<CompositeNodeCursor> entityCursorScan(KernelTransaction transaction, int[] labelIds) {
        var read = transaction.dataRead();
        read.prepareForLabelScans();
        List<Scan<NodeLabelIndexCursor>> scans = Arrays
            .stream(labelIds)
            .mapToObj(read::nodeLabelScan)
            .collect(Collectors.toList());
        return new CompositeNodeScan(scans);
    }

    @Override
    NodeReference cursorReference(KernelTransaction transaction, CompositeNodeCursor cursor) {
        return new MultipleNodeLabelIndexReference(
            cursor,
            transaction.dataRead(),
            Neo4jProxy.allocateNodeCursor(transaction.cursors(), transaction.pageCursorTracer())
        );
    }
}
