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

import com.carrotsearch.hppc.LongHashSet;
import com.carrotsearch.hppc.LongSet;
import org.immutables.builder.Builder;
import org.jetbrains.annotations.TestOnly;
import org.neo4j.graphalgo.utils.CheckedRunnable;
import org.neo4j.token.api.TokenConstants;
import org.neo4j.util.FeatureToggles;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.neo4j.graphalgo.core.GraphDimensions.IGNORE;
import static org.neo4j.kernel.impl.store.record.AbstractBaseRecord.NO_ID;

public final class NodesBatchBuffer extends RecordsBatchBuffer<NodeReference> {

    private static final AtomicBoolean SKIP_ORPHANS =
        new AtomicBoolean(FeatureToggles.flag(NodesBatchBuffer.class, "skipOrphans", false));

    @TestOnly
    public static synchronized <E extends Exception> void whileSkippingOrphans(CheckedRunnable<E> code) throws E {
        var skipOrphans = SKIP_ORPHANS.getAndSet(true);
        try {
            code.checkedRun();
        } finally {
            SKIP_ORPHANS.set(skipOrphans);
        }
    }

    private final LongSet nodeLabelIds;
    private final boolean hasLabelInformation;
    private final long[][] labelIds;
    private final boolean skipOrphans;

    // property ids, consecutive
    private final long[] properties;

    @Builder.Constructor
    NodesBatchBuffer(
        int capacity,
        Optional<LongSet> nodeLabelIds,
        Optional<Boolean> hasLabelInformation,
        Optional<Boolean> readProperty
    ) {
        super(capacity);
        this.nodeLabelIds = nodeLabelIds.orElseGet(LongHashSet::new);
        this.hasLabelInformation = hasLabelInformation.orElse(false);
        this.properties = readProperty.orElse(false) ? new long[capacity] : null;
        this.labelIds = new long[capacity][];
        this.skipOrphans = SKIP_ORPHANS.get();
    }

    @Override
    public void offer(final NodeReference record) {
        if (skipOrphans && record.relationshipReference() == NO_ID) {
            return;
        }
        if (nodeLabelIds.isEmpty()) {
            long propertiesReference = properties == null ? NO_ID : record.propertiesReference();
            add(record.nodeId(), propertiesReference, new long[]{TokenConstants.ANY_LABEL});
        } else {
            boolean atLeastOneLabelFound = false;
            var labels = record.labels();
            for (int i = 0; i < labels.length; i++) {
                long l = labels[i];
                if (!nodeLabelIds.contains(l) && !nodeLabelIds.contains(TokenConstants.ANY_LABEL)) {
                    labels[i] = IGNORE;
                } else {
                    atLeastOneLabelFound = true;
                }
            }
            if (atLeastOneLabelFound) {
                long propertiesReference = properties == null ? NO_ID : record.propertiesReference();
                add(record.nodeId(), propertiesReference, labels);
            }
        }
    }

    public void add(long nodeId, long propertiesIndex, long[] labels) {
        int len = length++;
        buffer[len] = nodeId;
        if (properties != null) {
            properties[len] = propertiesIndex;
        }
        if (labelIds != null) {
            labelIds[len] = labels;
        }
    }

    public long[] properties() {
        return this.properties;
    }

    public boolean hasLabelInformation() {
        return hasLabelInformation;
    }

    public long[][] labelIds() {
        return this.labelIds;
    }
}
