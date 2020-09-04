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
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.compat.GraphDatabaseApiProxy;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.ImmutableGraphCreateFromStoreConfig;
import org.neo4j.graphalgo.config.RelationshipWeightConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static java.util.Collections.singletonList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.graphalgo.ElementProjection.PROJECT_ALL;
import static org.neo4j.graphalgo.QueryRunner.runQuery;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.TestSupport.fromGdl;
import static org.neo4j.graphalgo.compat.MapUtil.map;
import static org.neo4j.graphalgo.config.AlgoBaseConfig.NODE_LABELS_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROJECTION_KEY;

public interface RelationshipWeightConfigTest<ALGORITHM extends Algorithm<ALGORITHM, RESULT>, CONFIG extends RelationshipWeightConfig & AlgoBaseConfig, RESULT> extends AlgoBaseProcTest<ALGORITHM, CONFIG, RESULT> {

    RelationshipProjections MULTI_RELATIONSHIPS_PROJECTION = RelationshipProjections.builder()
        .putProjection(
            RelationshipType.of("TYPE"),
            RelationshipProjection.builder()
                .type("TYPE")
                .properties(
                    PropertyMappings.of(
                        PropertyMapping.of("weight1", 0.0),
                        PropertyMapping.of("weight2", 1.0)
                    )
                )
                .build()
        )
        .putProjection(
            RelationshipType.of("TYPE1"),
            RelationshipProjection.builder()
                .type("TYPE1")
                .build()
        )
        .build();

    NodeProjections MULTI_NODES_PROJECTION = NodeProjections.builder()
        .putProjection(
            NodeLabel.of("Label"),
            NodeProjection.of("Label", PropertyMappings.of())
        )
        .putProjection(
            NodeLabel.of("Ignore"),
            NodeProjection.of("Ignore", PropertyMappings.of())
        )
        .build();


    String CREATE_QUERY = "CREATE" +
                          "  (x: Ignore)" +
                          ", (a: Label)" +
                          ", (b: Label)" +
                          ", (c: Label)" +
                          ", (y: Ignore)" +
                          ", (z: Ignore)" +
                          ", (a)-[:TYPE { weight1: 0.0, weight2: 1.0 }]->(b)" +
                          ", (a)-[:TYPE { weight2: 1.0 }]->(c)" +
                          ", (b)-[:TYPE { weight1: 0.0 }]->(c)" +
                          ", (c)-[:TYPE1 { weight1: 0.0 }]->(a)" +
                          ", (x)-[:TYPE]->(z)" +
                          ", (y)-[:TYPE]->(a)";

    @Test
    default void testDefaultRelationshipWeightPropertyIsNull() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.empty();
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.relationshipWeightProperty());
    }

    @Test
    default void testRelationshipWeightPropertyFromConfig() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(map("relationshipWeightProperty", "weight"));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertEquals("weight", config.relationshipWeightProperty());
    }

    @Test
    default void testEmptyRelationshipWeightPropertyValues() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(map("relationshipWeightProperty", null));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.relationshipWeightProperty());
    }

    @Test
    default void testTrimmedToNullRelationshipWeightProperty() {
        CypherMapWrapper mapWrapper = CypherMapWrapper.create(MapUtil.map("relationshipWeightProperty", "  "));
        CONFIG config = createConfig(createMinimalConfig(mapWrapper));
        assertNull(config.relationshipWeightProperty());
    }

    @Test
    default void testRelationshipWeightPropertyValidation() {
        runQuery(graphDb(), "CREATE ()-[:A {a: 1}]->()");
        List<String> relationshipProperties = singletonList("a");
        Map<String, Object> tempConfig = map(
            "relationshipWeightProperty", "foo",
            NODE_PROJECTION_KEY, PROJECT_ALL,
            RELATIONSHIP_PROJECTION_KEY, map(
                "A", map(
                    "properties", relationshipProperties
                )
            )
        );

        Map<String, Object> config = createMinimalConfig(CypherMapWrapper.create(tempConfig)).toMap();

        applyOnProcedure(proc -> {
            IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> proc.compute(config, Collections.emptyMap())
            );
            assertThat(e.getMessage(), containsString("foo"));
            assertThat(e.getMessage(), containsString("[a]"));
        });
    }

    @Test
    default void shouldFailWithInvalidRelationshipWeightProperty() {
        String loadedGraphName = "loadedGraph";
        GraphCreateConfig graphCreateConfig = emptyWithNameNative("", loadedGraphName);

        applyOnProcedure((proc) -> {
            GraphStore graphStore = graphLoader(graphCreateConfig).graphStore();

            GraphStoreCatalog.set(graphCreateConfig, graphStore);

            CypherMapWrapper mapWrapper = CypherMapWrapper.create(map(
                "relationshipWeightProperty",
                "___THIS_PROPERTY_SHOULD_NOT_EXIST___"
            ));
            Map<String, Object> configMap = createMinimalConfig(mapWrapper).toMap();
            String error = "Relationship weight property `___THIS_PROPERTY_SHOULD_NOT_EXIST___` not found in graph " +
                           "with relationship properties: [] in all relationship types: ['__ALL__']";
            assertMissingProperty(error, () -> proc.compute(
                loadedGraphName,
                configMap
            ));

            Map<String, Object> implicitConfigMap = createMinimalImplicitConfig(mapWrapper).toMap();
            assertMissingProperty(error, () -> proc.compute(
                implicitConfigMap,
                Collections.emptyMap()
            ));
        });
    }

    @Test
    default void shouldFailWithInvalidRelationshipWeightPropertyOnFilteredGraph() {
        runQuery(graphDb(), "MATCH (n) DETACH DELETE n");

        runQuery(graphDb(), "CREATE" +
                            "  (a:Node)" +
                            ", (b:Node)" +
                            ", (a)-[:Type]->(b)" +
                            ", (a)-[:Ignore {foo: 42}]->(b)");

        String loadedGraphName = "loadedGraph";

        GraphLoader graphLoader = new StoreLoaderBuilder()
            .api(graphDb())
            .graphName(loadedGraphName)
            .addRelationshipType("Type")
            .addRelationshipProjection(RelationshipProjection.builder()
                .type("Ignore")
                .addProperty("foo", "foo", DefaultValue.of(0))
                .build()
            ).build();

        GraphStoreCatalog.set(graphLoader.createConfig(), graphLoader.graphStore());

        applyOnProcedure((proc) -> {

            CypherMapWrapper mapWrapper = CypherMapWrapper.create(map(
                "relationshipWeightProperty", "foo",
                "relationshipTypes", List.of("Type")
            ));
            Map<String, Object> configMap = createMinimalConfig(mapWrapper).toMap();
            String error = "Relationship weight property `foo` not found in graph with relationship properties: [] in all relationship types: ['Type']";
            assertMissingProperty(error, () -> proc.compute(
                loadedGraphName,
                configMap
            ));

            Map<String, Object> implicitConfigMap = createMinimalImplicitConfig(mapWrapper).toMap();
            assertMissingProperty(error, () -> proc.compute(
                implicitConfigMap,
                Collections.emptyMap()
            ));
        });
    }

    @ParameterizedTest
    @CsvSource(value = {"weight1, 0.0", "weight2, 1.0"})
    default void testFilteringOnRelationshipPropertiesOnLoadedGraph(String propertyName, double expectedWeight) {
        String graphName = "foo";
        applyOnProcedure((proc) -> {
            loadExplicitGraphWithRelationshipWeights(graphName, MULTI_NODES_PROJECTION, MULTI_RELATIONSHIPS_PROJECTION);

            CypherMapWrapper weightConfig = CypherMapWrapper.create(map(
                "relationshipTypes", singletonList("TYPE"),
                "relationshipWeightProperty", propertyName
                )
            );

            CypherMapWrapper algoConfig = createMinimalConfigWithFilteredNodes(weightConfig);

            CONFIG config = proc.newConfig(Optional.of(graphName), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.of(graphName));

            Graph graph = proc.createGraph(configAndName);
            graph.forEachNode(nodeId -> {
                graph.forEachRelationship(nodeId, Double.NaN, (s, t, w) -> {
                    assertEquals(expectedWeight, w);
                    return true;
                });
                return true;
            });

        });
    }

    @ParameterizedTest
    @CsvSource(
        delimiter = ';',
        value = {
            "TYPE1; (:Label)-[]->(:Label), (:Label)",
            "TYPE; (c:Label)<--(a:Label)-->(b:Label)-->(c)",
            "*; (a:Label)-->(b:Label)-->(c:Label)-->(a)-->(c)"
        })
    default void testRunUnweightedOnWeightedMultiRelTypeGraph(String relType, String expectedGraph) {
        String weightedGraphName = "weightedGraph";
        applyOnProcedure((proc) -> {
            loadExplicitGraphWithRelationshipWeights(weightedGraphName, MULTI_NODES_PROJECTION, MULTI_RELATIONSHIPS_PROJECTION);

            CypherMapWrapper configWithoutRelWeight = CypherMapWrapper.create(map(
                "relationshipTypes",
                singletonList(relType)
            ));
            CypherMapWrapper algoConfig = createMinimalConfigWithFilteredNodes(configWithoutRelWeight);

            CONFIG config = proc.newConfig(Optional.of(weightedGraphName), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.of(weightedGraphName));

            Graph graph = proc.createGraph(configAndName);
            assertGraphEquals(fromGdl(expectedGraph), graph);
        });
    }

    @Test
    default void testRunUnweightedOnWeightedNoRelTypeGraph() {
        String noRelGraph = "weightedGraph";

        applyOnProcedure((proc) -> {
            RelationshipProjections relationshipProjections = RelationshipProjections
                .all()
                .addPropertyMappings(PropertyMappings.of(PropertyMapping.of("weight1", 1.0)));

            loadExplicitGraphWithRelationshipWeights(noRelGraph, MULTI_NODES_PROJECTION, relationshipProjections);

            CypherMapWrapper algoConfig = createMinimalConfigWithFilteredNodes(CypherMapWrapper.empty());

            CONFIG config = proc.newConfig(Optional.of(noRelGraph), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.of(noRelGraph));

            Graph graph = proc.createGraph(configAndName);
            assertGraphEquals(fromGdl("(a:Label)-->(b:Label)-->(c:Label)-->(a)-->(c)"), graph);
        });
    }

    @Test
    default void testRunUnweightedOnWeightedImplicitlyLoadedGraph() {
        runQuery(graphDb(), "MATCH (n) DETACH DELETE n");
        runQuery(graphDb(), CREATE_QUERY);

        String labelString = "Label";

        CypherMapWrapper weightConfig = CypherMapWrapper.create(map(
            NODE_PROJECTION_KEY, NodeProjections.builder()
                .putProjection(NodeLabel.of(labelString), NodeProjection.of(labelString, PropertyMappings.of()))
                .build(),
            RELATIONSHIP_PROJECTION_KEY, "*",
            "relationshipProperties", "weight1"
        ));
        CypherMapWrapper algoConfig = createMinimalConfig(weightConfig);

        applyOnProcedure((proc) -> {
            CONFIG config = proc.newConfig(Optional.empty(), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.empty());
            Graph graph = proc.createGraph(configAndName);
            assertGraphEquals(fromGdl("(a:Label)-->(b:Label)-->(c:Label)-->(a)-->(c)"), graph);
        });
    }

    @Test
    default void testFilteringOnRelTypesOnLoadedGraph() {
        String graphName = "foo";
        applyOnProcedure((proc) -> {
            loadExplicitGraphWithRelationshipWeights(graphName, MULTI_NODES_PROJECTION, MULTI_RELATIONSHIPS_PROJECTION);

            CypherMapWrapper weightConfig = CypherMapWrapper.create(MapUtil.map(
                "relationshipTypes", singletonList("TYPE"),
                "relationshipWeightProperty", "weight1"
            ));
            CypherMapWrapper algoConfig = createMinimalConfigWithFilteredNodes(weightConfig);

            CONFIG config = proc.newConfig(Optional.of(graphName), algoConfig);
            Pair<CONFIG, Optional<String>> configAndName = Tuples.pair(config, Optional.of(graphName));

            Graph graph = proc.createGraph(configAndName);
            assertGraphEquals(fromGdl("(a:Label)-[{w1: 0.0}]->(b:Label), (a:Label)-[{w1: 0.0}]->(c:Label), (b:Label)-[{w1: 0.0}]->(c:Label)"), graph);
        });
    }

    default void loadExplicitGraphWithRelationshipWeights(String graphName, NodeProjections nodeProjections, RelationshipProjections relationshipProjections) {
        GraphDatabaseAPI db = emptyDb();

        try {
            GraphDatabaseApiProxy.registerProcedures(db, GraphCreateProc.class);
        } catch (Exception ke) {}

        runQuery(db, CREATE_QUERY);

        GraphCreateConfig graphCreateConfig = ImmutableGraphCreateFromStoreConfig.builder()
            .graphName(graphName)
            .nodeProjections(nodeProjections)
            .relationshipProjections(relationshipProjections)
            .build();

        GraphStore graphStore = graphLoader(db, graphCreateConfig).graphStore();

        GraphStoreCatalog.set(graphCreateConfig, graphStore);
    }

    default CypherMapWrapper createMinimalConfigWithFilteredNodes(CypherMapWrapper config) {
        return createMinimalConfig(config).withEntry(NODE_LABELS_KEY, Collections.singletonList("Label"));
    }
}
