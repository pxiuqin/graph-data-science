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
package org.neo4j.graphalgo.catalog;

import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Map;
import java.util.stream.Stream;

import static org.neo4j.procedure.Mode.READ;

public class GraphListProc extends CatalogProc {

    private static final String NO_VALUE = "__NO_VALUE";
    private static final String DESCRIPTION = "Lists information about named graphs stored in the catalog.";

    @Procedure(name = "gds.graph.list", mode = READ)
    @Description(DESCRIPTION)
    public Stream<GraphInfoWithHistogram> list(@Name(value = "graphName", defaultValue = NO_VALUE) String graphName) {
        Stream<Map.Entry<GraphCreateConfig, GraphStore>> graphEntries = GraphStoreCatalog
            .getGraphStores(username())
            .entrySet()
            .stream();

        if (graphName != null && !graphName.equals(NO_VALUE)) {
            validateGraphName(graphName);

            // we should only list the provided graph
            graphEntries = graphEntries.filter(e -> e.getKey().graphName().equals(graphName));
        }

        return graphEntries.map(e -> {
            GraphCreateConfig graphCreateConfig = e.getKey();
            GraphStore graphStore = e.getValue();
            return GraphInfoWithHistogram.of(graphCreateConfig, graphStore);
        });
    }

}
