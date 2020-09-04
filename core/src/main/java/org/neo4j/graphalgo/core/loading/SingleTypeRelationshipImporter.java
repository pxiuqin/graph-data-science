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

import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipType;
import org.neo4j.graphalgo.api.IdMapping;
import org.neo4j.internal.kernel.api.CursorFactory;
import org.neo4j.internal.kernel.api.Read;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.memory.MemoryTracker;

import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Stream;

final class SingleTypeRelationshipImporter {

    private final RelationshipImporter.Imports imports;
    private final RelationshipImporter.PropertyReader propertyReader;
    private final RelationshipsBatchBuffer buffer;

    private SingleTypeRelationshipImporter(
            RelationshipImporter.Imports imports,
            RelationshipImporter.PropertyReader propertyReader,
            RelationshipsBatchBuffer buffer) {
        this.imports = imports;
        this.propertyReader = propertyReader;
        this.buffer = buffer;
    }

    RelationshipsBatchBuffer buffer() {
        return buffer;
    }

    long importRelationships() {
        return imports.importRelationships(buffer, propertyReader);
    }

    static class Builder {

        private final RelationshipType relationshipType;
        private final RelationshipProjection projection;
        private final RelationshipImporter importer;
        private final LongAdder relationshipCounter;
        private final int typeId;
        private final boolean validateRelationships;
        private final boolean loadProperties;

        Builder(
            RelationshipType relationshipType,
            RelationshipProjection projection,
            boolean loadProperties,
            int typeToken,
            RelationshipImporter importer,
            LongAdder relationshipCounter,
            boolean validateRelationships
        ) {
            this.relationshipType = relationshipType;
            this.projection = projection;
            this.typeId = typeToken;
            this.importer = importer;
            this.relationshipCounter = relationshipCounter;
            this.loadProperties = loadProperties && projection.properties().hasMappings();
            this.validateRelationships = validateRelationships;
        }

        RelationshipType relationshipType() {
            return relationshipType;
        }

        LongAdder relationshipCounter() {
            return relationshipCounter;
        }

        boolean loadProperties() {
            return this.loadProperties;
        }

        WithImporter loadImporter(boolean loadProperties) {
            RelationshipImporter.Imports imports = importer.imports(projection.orientation(), loadProperties);
            return new WithImporter(imports);
        }

        class WithImporter {
            private final RelationshipImporter.Imports imports;

            WithImporter(RelationshipImporter.Imports imports) {
                this.imports = imports;
            }

            Stream<Runnable> flushTasks() {
                return importer.flushTasks().stream();
            }

            SingleTypeRelationshipImporter withBuffer(
                IdMapping idMap,
                int bulkSize,
                RelationshipImporter.PropertyReader propertyReader
            ) {
                RelationshipsBatchBuffer buffer = new RelationshipsBatchBuffer(
                    idMap.cloneIdMapping(),
                    typeId,
                    bulkSize,
                    validateRelationships
                );
                return new SingleTypeRelationshipImporter(imports, propertyReader, buffer);
            }

            SingleTypeRelationshipImporter withBuffer(
                IdMapping idMap,
                int bulkSize,
                Read read,
                CursorFactory cursors,
                PageCursorTracer cursorTracer,
                MemoryTracker memoryTracker
            ) {
                RelationshipsBatchBuffer buffer = new RelationshipsBatchBuffer(
                    idMap.cloneIdMapping(),
                    typeId,
                    bulkSize,
                    validateRelationships
                );
                RelationshipImporter.PropertyReader propertyReader = loadProperties
                    ? importer.storeBackedPropertiesReader(cursors, read, cursorTracer, memoryTracker)
                    : (batch, batchLength, propertyKeyIds, defaultValues, aggregations, atLeastOnePropertyToLoad) -> new long[propertyKeyIds.length][0];
                return new SingleTypeRelationshipImporter(imports, propertyReader, buffer);
            }
        }
    }
}
