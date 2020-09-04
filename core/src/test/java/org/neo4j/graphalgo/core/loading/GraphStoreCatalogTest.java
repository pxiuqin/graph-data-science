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

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.TestSupport;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.extension.GdlExtension;
import org.neo4j.graphalgo.extension.GdlGraph;
import org.neo4j.graphalgo.extension.Inject;
import org.neo4j.graphalgo.gdl.GdlFactory;
import org.neo4j.kernel.database.DatabaseIdFactory;
import org.neo4j.kernel.database.NamedDatabaseId;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.extension.GdlSupportExtension.DATABASE_ID;

@GdlExtension
class GraphStoreCatalogTest {

    private static final String USER_NAME = "";
    private static final String GRAPH_NAME = "graph";
    private static final GraphCreateFromStoreConfig CONFIG = GraphCreateFromStoreConfig.emptyWithName(USER_NAME, GRAPH_NAME);

    @GdlGraph
    private static final String TEST_GRAPH = "()";

    @Inject
    private GraphStore graphStore;

    @Test
    void set() {
        assertFalse(GraphStoreCatalog.exists(USER_NAME, DATABASE_ID, GRAPH_NAME));
        GraphStoreCatalog.set(CONFIG, graphStore);
        assertTrue(GraphStoreCatalog.exists(USER_NAME, DATABASE_ID, GRAPH_NAME));
    }

    @Test
    void get() {
        GraphStoreCatalog.set(CONFIG, graphStore);
        GraphStoreWithConfig graphStoreWithConfig = GraphStoreCatalog.get(USER_NAME, DATABASE_ID, GRAPH_NAME);
        assertEquals(graphStore, graphStoreWithConfig.graphStore());
        assertEquals(CONFIG, graphStoreWithConfig.config());
    }

    @Test
    void remove() {
        GraphStoreCatalog.set(CONFIG, graphStore);
        assertTrue(GraphStoreCatalog.exists(USER_NAME, DATABASE_ID, GRAPH_NAME));
        GraphStoreCatalog.remove(USER_NAME, DATABASE_ID, GRAPH_NAME, graphStoreWithConfig -> {}, true);
        assertFalse(GraphStoreCatalog.exists(USER_NAME, DATABASE_ID, GRAPH_NAME));
    }

    @Test
    void graphStoresCount() {
        assertEquals(0, GraphStoreCatalog.graphStoresCount(DATABASE_ID));
        GraphStoreCatalog.set(CONFIG, graphStore);
        assertEquals(1, GraphStoreCatalog.graphStoresCount(DATABASE_ID));
        GraphStoreCatalog.remove(USER_NAME, DATABASE_ID, GRAPH_NAME, graphStoreWithConfig -> {}, true);
        assertEquals(0, GraphStoreCatalog.graphStoresCount(DATABASE_ID));
    }

    @Test
    void removeAllLoadedGraphs() {
        GraphStoreCatalog.set(CONFIG, graphStore);
        assertEquals(1, GraphStoreCatalog.graphStoresCount(DATABASE_ID));
        GraphStoreCatalog.removeAllLoadedGraphs();
        assertEquals(0, GraphStoreCatalog.graphStoresCount(DATABASE_ID));
    }

    @Test
    void getUnion() {
        GraphStoreCatalog.set(CONFIG, graphStore);
        var actual = GraphStoreCatalog.getUnion(USER_NAME, DATABASE_ID, GRAPH_NAME).get();
        TestSupport.assertGraphEquals(graphStore.getUnion(), actual);
    }

    @Test
    void multipleDatabaseIds() {
        GraphCreateFromStoreConfig config0 = GraphCreateFromStoreConfig.emptyWithName(USER_NAME, "graph0");
        GraphCreateFromStoreConfig config1 = GraphCreateFromStoreConfig.emptyWithName(USER_NAME, "graph1");

        NamedDatabaseId namedDatabaseId0 = DatabaseIdFactory.from("DB_0", UUID.fromString("0-0-0-0-0"));
        NamedDatabaseId namedDatabaseId1 = DatabaseIdFactory.from("DB_1", UUID.fromString("0-0-0-0-1"));

        GraphStore graphStore0 = GdlFactory.of("()-->()", namedDatabaseId0).build().graphStore();
        GraphStore graphStore1 = GdlFactory.of("()-->()-->()", namedDatabaseId1).build().graphStore();

        GraphStoreCatalog.set(config0, graphStore0);
        GraphStoreCatalog.set(config1, graphStore1);

        assertTrue(GraphStoreCatalog.exists(USER_NAME, namedDatabaseId0, "graph0"));
        assertFalse(GraphStoreCatalog.exists(USER_NAME, namedDatabaseId0, "graph1"));

        assertTrue(GraphStoreCatalog.exists(USER_NAME, namedDatabaseId1, "graph1"));
        assertFalse(GraphStoreCatalog.exists(USER_NAME, namedDatabaseId1, "graph0"));
    }
}
