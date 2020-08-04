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
package org.neo4j.graphalgo.config;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.QueryRunner;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.test.TestProc;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

class ConcurrencyConfigTest extends BaseProcTest {

    @BeforeEach
    void setupGraph() throws Exception {
        initDb(db, "'myG'");
    }

    private void initDb(GraphDatabaseAPI db, String graphName) throws Exception {
        registerProcedures(db, TestProc.class, GraphCreateProc.class);
        QueryRunner.runQuery(db, "CREATE (:A)");
        QueryRunner.runQuery(db, "CALL gds.graph.create(" + graphName + ", '*', '*')");
    }

    @AfterEach
    void tearDown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    @Test
    void shouldThrowOnTooHighConcurrency() {
        String query = "CALL gds.testProc.test('myG', {concurrency: 10, writeProperty: 'p'})";

        assertError(
            query,
            "The configured `concurrency` value is too high. " +
            "The maximum allowed `concurrency` value is 4 but 10 was configured."
        );
    }

    @Test
    void shouldThrowOnTooHighReadConcurrency() {
        String query = "CALL gds.graph.create('myG2', '*', '*', {readConcurrency: 9})";

        assertError(
            query,
            "The configured `readConcurrency` value is too high. " +
            "The maximum allowed `readConcurrency` value is 4 but 9 was configured."
        );
    }

    @Test
    void shouldThrowOnTooHighWriteConcurrency() {
        String query = "CALL gds.testProc.test('myG', {writeConcurrency: 12, writeProperty: 'p'})";

        assertError(
            query,
            "The configured `writeConcurrency` value is too high. " +
            "The maximum allowed `writeConcurrency` value is 4 but 12 was configured."
        );
    }
}
