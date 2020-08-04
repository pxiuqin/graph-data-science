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

import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.compat.GraphDatabaseApiProxy;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.ImmutableGraphCreateFromStoreConfig;
import org.neo4j.graphalgo.config.NodeWeightConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static java.util.Collections.emptyMap;
import static java.util.Collections.singletonList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.graphalgo.QueryRunner.runQuery;

public interface NodeWeightConfigTest<ALGORITHM extends Algorithm<ALGORITHM, RESULT>, CONFIG extends NodeWeightConfig & AlgoBaseConfig, RESULT> extends AlgoBaseProcTest<ALGORITHM, CONFIG, RESULT> {

    NodeProjections MULTI_PROPERTY_NODE_PROJECTION = NodeProjections.builder()
        .putProjection(
            NodeLabel.of("Label"),
            NodeProjection.builder()
                .label("Label")
                .properties(
                    PropertyMappings.of(
                        PropertyMapping.of("weight1", 0.0),
                        PropertyMapping.of("weight2", 1.0)
                    )
                )
                .build()
        ).build();

    @Test
    default void testDefaultNodeWeightPropertyIsNull() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.empty();
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.nodeWeightProperty());
    }

    @Test
    default void testTrimmedToNullNodeWeightProperty() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map("nodeWeightProperty", "  "));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.nodeWeightProperty());
    }

    @Test
    default void testNodeWeightPropertyFromConfig() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map("nodeWeightProperty", "weight"));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertEquals("weight", config.nodeWeightProperty());
    }

    @Test
    default void testEmptyNodeWeightPropertyValues() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map("nodeWeightProperty", null));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.nodeWeightProperty());
    }

    @Test
    default void testNodeWeightPropertyValidation() {
        runQuery(graphDb(), "CREATE (:A {a: 1})");
        Map<String, Object> tempConfig = MapUtil.map(
            "nodeWeightProperty", "foo",
            "nodeProjection", MapUtil.map(
                "A", MapUtil.map(
                    "properties", singletonList("a")
                )
            ),
            "relationshipProjection", "*"
        );

        Map<String, Object> config = createMinimalConfig(CypherMapWrapper.create(tempConfig)).toMap();

        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> {
                applyOnProcedure(proc -> proc.compute(config, emptyMap()));
            }
        );
        assertThat(e.getMessage(), containsString("foo"));
        assertThat(e.getMessage(), containsString("[a]"));
    }

    @Test
    default void shouldFailWithInvalidNodeWeightProperty() {
        String loadedGraphName = "loadedGraph";
        GraphCreateConfig graphCreateConfig = emptyWithNameNative("", loadedGraphName);

        applyOnProcedure((proc) -> {
            GraphStore graphStore = graphLoader(graphCreateConfig).graphStore();
            GraphStoreCatalog.set(graphCreateConfig, graphStore);

            CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map(
                "nodeWeightProperty",
                "___THIS_PROPERTY_SHOULD_NOT_EXIST___"
            ));
            Map<String, Object> configMap = createMinimalConfig(mapWrapper).toMap();
            String error = "Node weight property `___THIS_PROPERTY_SHOULD_NOT_EXIST___` not found in graph " +
                           "with node properties: [] in all node labels: ['__ALL__']";
            assertMissingProperty(error, () -> proc.compute(
                loadedGraphName,
                configMap
            ));

            Map<String, Object> implicitConfigMap = createMinimalImplicitConfig(mapWrapper).toMap();
            assertMissingProperty(error, () -> proc.compute(
                implicitConfigMap,
                emptyMap()
            ));
        });
    }

    @Test
    default void shouldFailWithInvalidNodeWeightPropertyOnFilteredGraph() {
        runQuery(graphDb(), "MATCH (n) DETACH DELETE n");

        runQuery(graphDb(), "CREATE" +
                            "  (a:Node)" +
                            ", (b:Ignore {foo: 42})" +
                            ", (a)-[:Type]->(b)");

        String loadedGraphName = "loadedGraph";

        GraphLoader graphLoader = new StoreLoaderBuilder()
            .api(graphDb())
            .graphName(loadedGraphName)
            .addNodeLabel("Node")
            .addNodeProjection(NodeProjection.builder()
                .label("Ignore")
                .addProperty("foo", "foo", 0)
                .build()
            ).build();

        GraphStoreCatalog.set(graphLoader.createConfig(), graphLoader.graphStore());

        applyOnProcedure((proc) -> {
            CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map(
                "nodeWeightProperty", "foo",
                "nodeLabels", List.of("Node")
            ));
            Map<String, Object> configMap = createMinimalConfig(mapWrapper).toMap();
            String error = "Node weight property `foo` not found in graph with node properties: [] in all node labels: ['Node']";
            assertMissingProperty(error, () -> proc.compute(
                loadedGraphName,
                configMap
            ));

            Map<String, Object> implicitConfigMap = createMinimalImplicitConfig(mapWrapper).toMap();
            assertMissingProperty(error, () -> proc.compute(
                implicitConfigMap,
                emptyMap()
            ));
        });
    }

    @ParameterizedTest
    @CsvSource(value = {"weight1, 0.0", "weight2, 1.0"})
    default void testFilteringOnNodePropertiesOnLoadedGraph(String propertyName, double expectedWeight) {
        String graphName = "foo";
        applyOnProcedure((proc) -> {
            loadExplicitGraphWithNodeWeights(graphName, MULTI_PROPERTY_NODE_PROJECTION);

            CypherMapWrapper weightConfig = CypherMapWrapper.create(MapUtil.map("nodeWeightProperty", propertyName));
            CypherMapWrapper algoConfig = createMinimalConfig(weightConfig);
            CONFIG config = proc.newConfig(Optional.of(graphName), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.of(graphName));

            Graph graph = proc.createGraph(configAndName);
            graph.forEachNode(nodeId -> {
                assertEquals(expectedWeight, graph.nodeProperties(propertyName).nodeProperty(nodeId));
                return true;
            });

        });
    }

    default void loadExplicitGraphWithNodeWeights(String graphName, NodeProjections nodeProjections) {
        GraphDatabaseAPI db = emptyDb();

        try {
            GraphDatabaseApiProxy.registerProcedures(db, GraphCreateProc.class);
        } catch (Exception ke) {}

        String createQuery = "CREATE" +
                             "  (a: Label { weight1: 0.0, weight2: 1.0 })" +
                             ", (b: Label { weight2: 1.0 })" +
                             ", (c: Label { weight1: 0.0 })";

        runQuery(db, createQuery);

        GraphCreateConfig graphCreateConfig = ImmutableGraphCreateFromStoreConfig.builder()
            .graphName(graphName)
            .nodeProjections(nodeProjections)
            .relationshipProjections(RelationshipProjections.all())
            .build();

        GraphStore graphStore = graphLoader(db, graphCreateConfig).graphStore();

        GraphStoreCatalog.set(graphCreateConfig, graphStore);
    }
}
