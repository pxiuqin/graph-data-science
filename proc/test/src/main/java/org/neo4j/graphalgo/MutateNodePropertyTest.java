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

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.MutateConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;

import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.fail;
import static org.neo4j.graphalgo.QueryRunner.runQuery;
import static org.neo4j.graphalgo.QueryRunner.runQueryWithRowConsumer;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public interface MutateNodePropertyTest<ALGORITHM extends Algorithm<ALGORITHM, RESULT>, CONFIG extends MutateConfig & AlgoBaseConfig, RESULT>
    extends MutateProcTest<ALGORITHM, CONFIG, RESULT> {

    @Test
    @Override
    default void testWriteBackGraphMutationOnFilteredGraph() {
        runQuery(graphDb(), "MATCH (n) DETACH DELETE n");
        GraphStoreCatalog.removeAllLoadedGraphs();

        runQuery(graphDb(), "CREATE (a1: A), (a2: A), (b: B), (a1)-[:REL]->(a2)");
        String graphName = "myGraph";

        StoreLoaderBuilder storeLoaderBuilder = new StoreLoaderBuilder()
            .api(graphDb())
            .graphName(graphName)
            .addNodeLabels("A", "B");

        relationshipProjections().projections().forEach((relationshipType, projection)->
            storeLoaderBuilder.putRelationshipProjectionsWithIdentifier(relationshipType.name(), projection));

        GraphLoader loader = storeLoaderBuilder.build();
        GraphStoreCatalog.set(loader.createConfig(), loader.graphStore());

        applyOnProcedure(procedure ->
            getProcedureMethods(procedure)
                .filter(procedureMethod -> getProcedureMethodName(procedureMethod).endsWith(".mutate"))
                .forEach(mutateMethod -> {
                    CypherMapWrapper filterConfig = CypherMapWrapper.empty().withEntry(
                        "nodeLabels",
                        Collections.singletonList("A")
                    );

                    Map<String, Object> config = createMinimalConfig(filterConfig).toMap();
                    try {
                        mutateMethod.invoke(procedure, graphName, config);
                    } catch (IllegalAccessException | InvocationTargetException e) {
                        fail(e);
                    }
                })
        );

        String graphWriteQuery =
            "CALL gds.graph.writeNodeProperties(" +
            "   $graph, " +
            "   [$property]" +
            ") YIELD writeMillis, graphName, nodeProperties, propertiesWritten";

        runQuery(graphDb(), graphWriteQuery, Map.of("graph", graphName, "property", mutateProperty()));

        String checkNeo4jGraphNegativeQuery = formatWithLocale("MATCH (n:B) RETURN n.%s AS property", mutateProperty());

        runQueryWithRowConsumer(
            graphDb(),
            checkNeo4jGraphNegativeQuery,
            Map.of(),
            ((transaction, resultRow) -> assertNull(resultRow.get("property")))
        );

        String checkNeo4jGraphPositiveQuery = formatWithLocale("MATCH (n:A) RETURN n.%s AS property", mutateProperty());

        runQueryWithRowConsumer(
            graphDb(),
            checkNeo4jGraphPositiveQuery,
            Map.of(),
            ((transaction, resultRow) -> assertNotNull(resultRow.get("property")))
        );
    }

    @Override
    default String failOnExistingTokenMessage() {
        return formatWithLocale(
            "Node property `%s` already exists in the in-memory graph.",
            mutateProperty()
        );
    }
}
