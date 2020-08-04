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

import org.neo4j.graphalgo.api.GraphLoaderContext;
import org.neo4j.graphalgo.core.GraphDimensions;
import org.neo4j.graphalgo.core.SecureTransaction;
import org.neo4j.graphalgo.core.loading.InternalImporter.ImportResult;
import org.neo4j.graphalgo.core.utils.paged.AllocationTracker;
import org.neo4j.logging.Log;
import org.neo4j.util.FeatureToggles;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.neo4j.graphalgo.core.loading.AbstractRecordBasedScanner.DEFAULT_PREFETCH_SIZE;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.humanReadable;

public abstract class ScanningRecordsImporter<Record, T> {

    private static final BigInteger A_BILLION = BigInteger.valueOf(1_000_000_000L);

    private static final boolean USE_KERNEL_CURSORS_DEFAULT_SETTING = true;
    private static final boolean USE_KERNEL_CURSORS_FLAG = FeatureToggles.flag(
        ScanningRecordsImporter.class,
        "useKernelCursors",
        USE_KERNEL_CURSORS_DEFAULT_SETTING
    );
    protected static final AtomicBoolean USE_KERNEL_CURSORS = new AtomicBoolean(USE_KERNEL_CURSORS_FLAG);

    private final StoreScanner.Factory<Record> factory;
    private final String label;
    private final ExecutorService threadPool;

    protected final SecureTransaction transaction;
    protected final GraphDimensions dimensions;
    protected final AllocationTracker tracker;
    protected final int concurrency;

    public ScanningRecordsImporter(
        StoreScanner.Factory<Record> factory,
        String label,
        GraphLoaderContext loadingContext,
        GraphDimensions dimensions,
        int concurrency
    ) {
        this.factory = factory;
        this.label = label;
        this.transaction = loadingContext.transaction();
        this.dimensions = dimensions;
        this.threadPool = loadingContext.executor();
        this.tracker = loadingContext.tracker();
        this.concurrency = concurrency;
    }

    public final T call(Log log) {
        long nodeCount = dimensions.nodeCount();
        final ImportSizing sizing = ImportSizing.of(concurrency, nodeCount);
        int numberOfThreads = sizing.numberOfThreads();

        try (StoreScanner<Record> scanner = factory.newScanner(DEFAULT_PREFETCH_SIZE, transaction)) {
            log.debug("%s Store Scan: Start using %s", label, scanner.getClass().getSimpleName());

            InternalImporter.CreateScanner creator = creator(nodeCount, sizing, scanner);
            InternalImporter importer = new InternalImporter(numberOfThreads, creator);
            ImportResult importResult = importer.runImport(threadPool);

            long requiredBytes = scanner.storeSize();
            long recordsImported = importResult.recordsImported;
            long propertiesImported = importResult.propertiesImported;
            BigInteger bigNanos = BigInteger.valueOf(importResult.tookNanos);
            double tookInSeconds = new BigDecimal(bigNanos)
                    .divide(new BigDecimal(A_BILLION), 9, RoundingMode.CEILING)
                    .doubleValue();
            long bytesPerSecond = A_BILLION
                .multiply(BigInteger.valueOf(requiredBytes))
                .divide(bigNanos)
                .longValueExact();

            log.info(
                    "%s Store Scan: Imported %,d records and %,d properties from %s (%,d bytes); took %.3f s, %,.2f %1$ss/s, %s/s (%,d bytes/s) (per thread: %,.2f %1$ss/s, %s/s (%,d bytes/s))",
                    label,
                    recordsImported,
                    propertiesImported,
                    humanReadable(requiredBytes),
                    requiredBytes,
                    tookInSeconds,
                    (double) recordsImported / tookInSeconds,
                    humanReadable(bytesPerSecond),
                    bytesPerSecond,
                    (double) recordsImported / tookInSeconds / numberOfThreads,
                    humanReadable(bytesPerSecond / numberOfThreads),
                    bytesPerSecond / numberOfThreads
            );
        }

        return build();
    }

    public abstract InternalImporter.CreateScanner creator(
        long nodeCount,
        ImportSizing sizing,
        StoreScanner<Record> scanner
    );

    public abstract T build();
}
