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
package org.neo4j.graphalgo;


import org.intellij.lang.annotations.Language;
import org.neo4j.graphalgo.core.EnterpriseLicensingExtension;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.extension.ExtensionCallback;
import org.neo4j.test.extension.ImpermanentDbmsExtension;
import org.neo4j.test.extension.Inject;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

import static java.util.Collections.emptyMap;

@ImpermanentDbmsExtension(configurationCallback = "configuration")
public abstract class BaseTest {

    @Inject
    public GraphDatabaseAPI db;

    @ExtensionCallback
    protected void configuration(TestDatabaseManagementServiceBuilder builder) {
        builder.impermanent();
        builder.noOpSystemGraphInitializer();
        builder.addExtension(new EnterpriseLicensingExtension());
    }

    protected long clearDb() {
        var deletedNodes = new AtomicLong();
        runQueryWithRowConsumer("MATCH (n) DETACH DELETE n RETURN count(n)",
            row -> deletedNodes.set(row.getNumber("count(n)").longValue()));
        return deletedNodes.get();
    }

    protected void runQueryWithRowConsumer(
        @Language("Cypher") String query,
        Consumer<Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(db, query, check);
    }

    protected void runQueryWithRowConsumer(
        @Language("Cypher") String query,
        BiConsumer<Transaction, Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(db, query, emptyMap(), check);
    }

    protected void runQueryWithRowConsumer(
        @Language("Cypher") String query,
        Map<String, Object> params,
        Consumer<Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(db, query, params, discardTx(check));
    }

    protected void runQueryWithRowConsumer(
        GraphDatabaseService localDb,
        @Language("Cypher") String query,
        Map<String, Object> params,
        Consumer<Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(localDb, query, params, discardTx(check));
    }

    protected void runQueryWithRowConsumer(
        GraphDatabaseService localDb,
        @Language("Cypher") String query,
        Consumer<Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(localDb, query, emptyMap(), discardTx(check));
    }

    protected void runQueryWithRowConsumer(
        String username,
        @Language("Cypher") String query,
        Consumer<Result.ResultRow> check
    ) {
        QueryRunner.runQueryWithRowConsumer(db, username, query, emptyMap(), discardTx(check));
    }

    protected void runQuery(
        String username,
        @Language("Cypher") String query,
        Map<String, Object> params
    ) {
        QueryRunner.runQuery(db, username, query, params);
    }

    protected void runQuery(
        GraphDatabaseService db,
        @Language("Cypher") String query,
        Map<String, Object> params
    ) {
        QueryRunner.runQuery(db, query, params);
    }

    protected void runQuery(@Language("Cypher") String query) {
        QueryRunner.runQuery(db, query);
    }

    protected void runQuery(
        @Language("Cypher") String query,
        Map<String, Object> params
    ) {
        QueryRunner.runQuery(db, query, params);
    }

    protected <T> T runQuery(
        @Language("Cypher") String query,
        Function<Result, T> resultFunction
    ) {
        return QueryRunner.runQuery(db, query, emptyMap(), resultFunction);
    }

    protected <T> T runQuery(
        @Language("Cypher") String query,
        Map<String, Object> params,
        Function<Result, T> resultFunction
    ) {
        return QueryRunner.runQuery(db, query, params, resultFunction);
    }

    protected <T> T runQuery(
        GraphDatabaseService db,
        @Language("Cypher") String query,
        Map<String, Object> params,
        Function<Result, T> resultFunction
    ) {
        return QueryRunner.runQuery(db, query, params, resultFunction);
    }

    protected void runQueryWithResultConsumer(
        @Language("Cypher") String query,
        Map<String, Object> params,
        Consumer<Result> check
    ) {
        QueryRunner.runQueryWithResultConsumer(
            db,
            query,
            params,
            check
        );
    }

    protected void runQueryWithResultConsumer(
        @Language("Cypher") String query,
        Consumer<Result> check
    ) {
        QueryRunner.runQueryWithResultConsumer(
            db,
            query,
            emptyMap(),
            check
        );
    }

    private static BiConsumer<Transaction, Result.ResultRow> discardTx(Consumer<Result.ResultRow> check) {
        return (tx, row) -> check.accept(row);
    }
}
