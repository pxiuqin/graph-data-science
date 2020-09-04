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

import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.neo4j.graphalgo.TestSupport.AllGraphStoreFactoryTypesTest;
import org.neo4j.graphalgo.api.GraphStore;
import org.neo4j.graphalgo.api.ImmutableGraphLoaderContext;
import org.neo4j.graphalgo.compat.GraphDatabaseApiProxy;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.GraphCreateConfig;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.config.GraphCreateFromStoreConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.GdsEdition;
import org.neo4j.graphalgo.core.GraphLoader;
import org.neo4j.graphalgo.core.ImmutableGraphLoader;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.internal.kernel.api.procs.ProcedureCallContext;
import org.neo4j.internal.kernel.api.security.AuthSubject;
import org.neo4j.kernel.database.NamedDatabaseId;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.procedure.Procedure;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.neo4j.graphalgo.BaseProcTest.anonymousGraphConfig;
import static org.neo4j.graphalgo.NodeLabel.ALL_NODES;
import static org.neo4j.graphalgo.QueryRunner.runQuery;
import static org.neo4j.graphalgo.RelationshipType.ALL_RELATIONSHIPS;
import static org.neo4j.graphalgo.TestSupport.FactoryType.CYPHER;
import static org.neo4j.graphalgo.compat.GraphDatabaseApiProxy.newKernelTransaction;
import static org.neo4j.graphalgo.config.GraphCreateConfig.IMPLICIT_GRAPH_NAME;
import static org.neo4j.graphalgo.config.GraphCreateConfig.READ_CONCURRENCY_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.ALL_NODES_QUERY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.ALL_RELATIONSHIPS_QUERY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.NODE_QUERY_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.RELATIONSHIP_QUERY_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROJECTION_KEY;
import static org.neo4j.graphalgo.utils.ExceptionUtil.rootCause;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

/**
 * Base test that should be used for every algorithm procedure.
 * This test assumes that the implementing test method populates
 * the database returned by {@link AlgoBaseProcTest#graphDb} and
 * clears the data after each test.
 */
public interface AlgoBaseProcTest<ALGORITHM extends Algorithm<ALGORITHM, RESULT>, CONFIG extends AlgoBaseConfig, RESULT>
    extends GraphCreateConfigSupport
{

    String TEST_USERNAME = AuthSubject.ANONYMOUS.username();

    @AfterEach
    default void removeAllLoadedGraphs() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    Class<? extends AlgoBaseProc<ALGORITHM, RESULT, CONFIG>> getProcedureClazz();

    String createQuery();

    GraphDatabaseAPI graphDb();

    default NamedDatabaseId namedDatabaseId() {
        return graphDb().databaseId();
    }

    default GraphDatabaseAPI emptyDb() {
        GraphDatabaseAPI db = graphDb();
        runQuery(db, "MATCH (n) DETACH DELETE n");
        return db;
    }

    CONFIG createConfig(CypherMapWrapper mapWrapper);

    void assertResultEquals(RESULT result1, RESULT result2);

    default CypherMapWrapper createMinimalConfig(CypherMapWrapper mapWrapper) {
        return mapWrapper;
    }

    default CypherMapWrapper createMinimalImplicitConfig(CypherMapWrapper mapWrapper) {
        return createMinimalConfig(CypherMapWrapper.create(anonymousGraphConfig(mapWrapper.toMap())));
    }

    default void applyOnProcedure(Consumer<? super AlgoBaseProc<ALGORITHM, RESULT, CONFIG>> func) {
        try (GraphDatabaseApiProxy.Transactions transactions = newKernelTransaction(graphDb())) {
            AlgoBaseProc<ALGORITHM, RESULT, CONFIG> proc;
            try {
                proc = getProcedureClazz().getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException("Could not instantiate Procedure Class " + getProcedureClazz().getSimpleName());
            }

            proc.transaction = transactions.ktx();
            proc.api = graphDb();
            proc.callContext = ProcedureCallContext.EMPTY;
            proc.log = new TestLog();

            func.accept(proc);
        }
    }

    @Test
    default void testImplicitGraphCreateFromStoreConfig() {
        CypherMapWrapper wrapper = createMinimalConfig(CypherMapWrapper.create(MapUtil.map(
            NODE_PROJECTION_KEY, Collections.singletonList("*"),
            RELATIONSHIP_PROJECTION_KEY, relationshipProjections()
        )));
        applyOnProcedure(proc -> {
            CONFIG config = proc.newConfig(Optional.empty(), wrapper);
            assertEquals(Optional.empty(), config.graphName(), "Graph name should be empty.");
            Optional<GraphCreateConfig> maybeGraphCreateConfig = config.implicitCreateConfig();
            assertTrue(maybeGraphCreateConfig.isPresent(), "Config should contain a GraphCreateConfig.");
            GraphCreateConfig actual = maybeGraphCreateConfig.get();
            assertTrue(
                actual instanceof GraphCreateFromStoreConfig,
                formatWithLocale("GraphCreateConfig should be %s.", GraphCreateFromStoreConfig.class.getSimpleName())
            );

            GraphCreateFromStoreConfig storeConfig = (GraphCreateFromStoreConfig) actual;

            NodeProjections expectedNodeProjections = expectedNodeProjections();
            RelationshipProjections expectedRelationshipProjections = relationshipProjections();

            assertEquals(expectedNodeProjections, storeConfig.nodeProjections());
            assertEquals(expectedRelationshipProjections, storeConfig.relationshipProjections());
            assertEquals(IMPLICIT_GRAPH_NAME, storeConfig.graphName());
            assertEquals(TEST_USERNAME, storeConfig.username());
        });
    }

    default NodeProjections expectedNodeProjections() {
        return NodeProjections
            .builder()
            .putProjection(ALL_NODES, NodeProjection.all())
            .build();
    }

    @Test
    default void testImplicitGraphCreateFromCypherConfig() {
        long concurrency = 2;
        Map<String, Object> tempConfig = MapUtil.map(
            NODE_QUERY_KEY, ALL_NODES_QUERY,
            RELATIONSHIP_QUERY_KEY, ALL_RELATIONSHIPS_QUERY,
            "concurrency", concurrency
        );
        CypherMapWrapper wrapper = createMinimalConfig(CypherMapWrapper.create(tempConfig));

        applyOnProcedure(proc -> {
            CONFIG config = proc.newConfig(Optional.empty(), wrapper);
            assertEquals(Optional.empty(), config.graphName(), "Graph name should be empty.");
            Optional<GraphCreateConfig> maybeGraphCreateConfig = config.implicitCreateConfig();
            assertTrue(maybeGraphCreateConfig.isPresent(), "Config should contain a GraphCreateConfig.");
            assertTrue(
                maybeGraphCreateConfig.get() instanceof GraphCreateFromCypherConfig,
                formatWithLocale("GraphCreateConfig should be %s.", GraphCreateFromCypherConfig.class.getSimpleName()));

            GraphCreateFromCypherConfig actualConfig = (GraphCreateFromCypherConfig) maybeGraphCreateConfig.get();

            assertEquals(ALL_NODES_QUERY, actualConfig.nodeQuery());
            assertEquals(ALL_RELATIONSHIPS_QUERY, actualConfig.relationshipQuery());
            assertEquals(IMPLICIT_GRAPH_NAME, actualConfig.graphName());
            assertEquals(TEST_USERNAME, actualConfig.username());
            assertEquals(concurrency, actualConfig.readConcurrency());
        });
    }

    default void assertMissingProperty(String error, Runnable runnable) {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            runnable::run
        );
        assertTrue(exception.getMessage().contains(error));
    }

    @AllGraphStoreFactoryTypesTest
    default void testRunOnLoadedGraph(TestSupport.FactoryType factoryType) {
        String loadedGraphName = "loadedGraph";
        GraphCreateConfig graphCreateConfig = factoryType == CYPHER
            ? withNameAndRelationshipProjections("", loadedGraphName, relationshipProjections())
            : withNameAndRelationshipQuery("", loadedGraphName, relationshipQuery());

        applyOnProcedure((proc) -> {
            GraphStore graphStore = graphLoader(graphCreateConfig).graphStore();
            GraphStoreCatalog.set(
                graphCreateConfig,
                graphStore
            );
            Map<String, Object> configMap = createMinimalConfig(CypherMapWrapper.empty()).toMap();
            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultOnLoadedGraph = proc.compute(
                loadedGraphName,
                configMap
            );

            Map<String, Object> implicitConfigMap = createMinimalImplicitConfig(CypherMapWrapper.empty()).toMap();
            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultOnImplicitGraph = proc.compute(
                implicitConfigMap,
                Collections.emptyMap()
            );

            assertResultEquals(resultOnImplicitGraph.result(), resultOnLoadedGraph.result());
        });
    }

    @Test
    default void testRunOnImplicitlyLoadedGraph() {
        Map<String, Object> cypherConfig = createMinimalConfig(CypherMapWrapper.create(MapUtil.map(
            NODE_QUERY_KEY, ALL_NODES_QUERY,
            RELATIONSHIP_QUERY_KEY, relationshipQuery()
        ))).toMap();

        Map<String, Object> storeConfig = createMinimalConfig(CypherMapWrapper.create(MapUtil.map(
            NODE_PROJECTION_KEY, Collections.singletonList("*"),
            RELATIONSHIP_PROJECTION_KEY, relationshipProjections()
        ))).toMap();

        applyOnProcedure((proc) -> {

            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultOnImplicitGraphFromCypher = proc.compute(
                cypherConfig,
                Collections.emptyMap()
            );

            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultOnImplicitGraphFromStore = proc.compute(
                storeConfig,
                Collections.emptyMap()
            );

            assertResultEquals(resultOnImplicitGraphFromCypher.result(), resultOnImplicitGraphFromStore.result());
        });
    }

    @Test
    default void useReadConcurrencyWhenSetOnImplicitlyLoadedGraph() {
        CypherMapWrapper config = createMinimalConfig(
            CypherMapWrapper.create(
                Map.of(
                    NODE_PROJECTION_KEY, NodeProjections.ALL,
                    RELATIONSHIP_PROJECTION_KEY, relationshipProjections(),
                    READ_CONCURRENCY_KEY, 2
                )
            )
        );

        applyOnProcedure((proc) -> {
            CONFIG algoConfig = proc.newConfig(Optional.empty(), config);
            assertTrue(algoConfig.implicitCreateConfig().isPresent());
            assertEquals(2, algoConfig.implicitCreateConfig().get().readConcurrency());
        });
    }

    default RelationshipProjections relationshipProjections() {
        return RelationshipProjections.ALL;
    }

    default String relationshipQuery() {
        return ALL_RELATIONSHIPS_QUERY;
    }

    @AllGraphStoreFactoryTypesTest
    default void testRunMultipleTimesOnLoadedGraph(TestSupport.FactoryType factoryType) {
        String loadedGraphName = "loadedGraph";
        GraphCreateConfig graphCreateConfig = factoryType == CYPHER
            ? emptyWithNameCypher(TEST_USERNAME, loadedGraphName)
            : withNameAndRelationshipProjections(TEST_USERNAME, loadedGraphName, relationshipProjections());

        applyOnProcedure((proc) -> {
            GraphStoreCatalog.set(
                graphCreateConfig,
                graphLoader(graphCreateConfig).graphStore()
            );
            Map<String, Object> configMap = createMinimalConfig(CypherMapWrapper.empty()).toMap();
            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultRun1 = proc.compute(loadedGraphName, configMap);
            AlgoBaseProc.ComputationResult<?, RESULT, CONFIG> resultRun2 = proc.compute(loadedGraphName, configMap);

            assertResultEquals(resultRun1.result(), resultRun2.result());
        });
    }

    @Test
    default void testRunOnEmptyGraph() {
        runQuery(graphDb(), "MATCH (n) DETACH DELETE n");

        applyOnProcedure((proc) -> {
            getWriteAndStreamProcedures(proc)
                .forEach(method -> {
                    Map<String, Object> configMap = createMinimalImplicitConfig(CypherMapWrapper.empty()).toMap();

                    try {
                        Stream<?> result = (Stream<?>) method.invoke(proc, configMap, Collections.emptyMap());

                        if (getProcedureMethodName(method).endsWith("stream")) {
                            assertEquals(0, result.count(), "Stream result should be empty.");
                        } else {
                            assertEquals(1, result.count());
                        }

                    } catch (IllegalAccessException | InvocationTargetException e) {
                        fail(e);
                    }
                });
        });
    }

    @Test
    default void testNonConsecutiveIds() {
        RESULT originalResult = loadGraphAndRunCompute("originalGraph");

        createRandomNodes();
        emptyDb();
        runQuery(graphDb(), createQuery());

        RESULT reloadedResult = loadGraphAndRunCompute("reloadedGraph");

        assertResultEquals(originalResult, reloadedResult);
    }

    default void loadGraph(String graphName){
        runQuery(
            graphDb(),
            GdsCypher.call()
                .loadEverything()
                .graphCreate(graphName)
                .yields()
        );
    }

    private RESULT loadGraphAndRunCompute(String graphName) {
        loadGraph(graphName);

        AtomicReference<AlgoBaseProc.ComputationResult<?, RESULT, CONFIG>> computationResult = new AtomicReference<>();
        applyOnProcedure((proc) -> {
            Map<String, Object> configMap = createMinimalConfig(CypherMapWrapper.empty()).toMap();
            computationResult.set(proc.compute(graphName, configMap));
        });
        return computationResult.get().result();
    }

    private void createRandomNodes() {
        runQuery(
            graphDb(),
            "UNWIND range(1,$size) as _ CREATE (:Fake)",
            Map.of("size", new Random().nextInt(1000))
        );
    }

    @Test
    default void testFailOnMissingNodeLabel() {
        applyOnProcedure((proc) -> {
            getWriteAndStreamProcedures(proc)
                .forEach(method -> {
                    String missingLabel = "___THIS_LABEL_SHOULD_NOT_EXIST___";
                    Map<String, Object> tempConfig = MapUtil.map(NODE_PROJECTION_KEY, Collections.singletonList(missingLabel));

                    Map<String, Object> configMap = createMinimalImplicitConfig(CypherMapWrapper.create(tempConfig)).toMap();

                    Exception ex = assertThrows(
                        Exception.class,
                        () -> method.invoke(proc, configMap, Collections.emptyMap())
                    );
                    Throwable rootCause = rootCause(ex);
                    assertEquals(IllegalArgumentException.class, rootCause.getClass());
                    assertThat(
                        rootCause.getMessage(),
                        containsString(formatWithLocale(
                            "Invalid node projection, one or more labels not found: '%s'",
                            missingLabel
                        ))
                    );
                });
        });
    }

    @Test
    default void shouldThrowWhenTooManyCoresOnLimited() {
        GdsEdition.instance().setToCommunityEdition();
        applyOnProcedure((proc) ->
            getWriteAndStreamProcedures(proc).forEach(method -> {
                Map<String, Object> configMap = createMinimalImplicitConfig(CypherMapWrapper.create(MapUtil.map(
                    "concurrency",
                    10
                ))).toMap();

                InvocationTargetException ex = assertThrows(
                    InvocationTargetException.class,
                    () -> method.invoke(proc, configMap, Collections.emptyMap())
                );
                assertEquals(IllegalArgumentException.class, ex.getCause().getClass());
                assertThat(ex.getCause().getMessage(), containsString("The configured `readConcurrency` value is too high"));
            })
        );
    }

    @Test
    default void shouldAllowManyCoresOnUnlimited() {
        GdsEdition.instance().setToEnterpriseEdition();
        applyOnProcedure((proc) ->
            getWriteAndStreamProcedures(proc).forEach(method -> {
                Map<String, Object> configMap = createMinimalImplicitConfig(CypherMapWrapper.create(MapUtil.map("concurrency", 78))).toMap();

                assertDoesNotThrow(() -> method.invoke(proc, configMap, Collections.emptyMap()));
            })
        );
    }

    @Test
    default void testFailOnMissingRelationshipType() {
        applyOnProcedure((proc) -> {
            getWriteAndStreamProcedures(proc)
                .forEach(method -> {
                    String missingRelType = "___THIS_REL_TYPE_SHOULD_NOT_EXIST___";
                    Map<String, Object> tempConfig = MapUtil.map(RELATIONSHIP_PROJECTION_KEY, Collections.singletonList(missingRelType));

                    Map<String, Object> configMap = createMinimalImplicitConfig(CypherMapWrapper.create(tempConfig)).toMap();

                    Exception ex = assertThrows(
                        Exception.class,
                        () -> method.invoke(proc, configMap, Collections.emptyMap())
                    );
                    Throwable rootCause = rootCause(ex);
                    assertEquals(IllegalArgumentException.class, rootCause.getClass());
                    assertThat(
                        rootCause.getMessage(),
                        containsString(formatWithLocale(
                            "Invalid relationship projection, one or more relationship types not found: '%s'",
                            missingRelType
                        ))
                    );
                });
        });
    }

    @Test
    default void failOnImplicitLoadingWithAlteringNodeQuery() {
        Map<String, Object> config = createMinimalConfig(CypherMapWrapper.create(MapUtil.map(
            NODE_QUERY_KEY, "MATCH (n) SET n.name='foo' RETURN id(n) AS id",
            RELATIONSHIP_QUERY_KEY, ALL_RELATIONSHIPS_QUERY
        ))).toMap();

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> applyOnProcedure((proc) -> proc.compute(
                config,
                Collections.emptyMap()
            ))
        );

        assertThat(ex.getMessage(), containsString("Query must be read only. Query: "));
    }

    // NOTE: this test needs at least one relationship in order to pass
    @Test
    default void failOnImplicitLoadingWithAlteringRelationshipQuery() {
        Map<String, Object> config = createMinimalConfig(CypherMapWrapper.create(MapUtil.map(
            NODE_QUERY_KEY, ALL_NODES_QUERY,
            RELATIONSHIP_QUERY_KEY, "MATCH (s)-->(t) SET s.foo=false RETURN id(s) AS source, id(t) as target"
        ))).toMap();

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> applyOnProcedure((proc) -> proc.compute(
                config,
                Collections.emptyMap()
            ))
        );

        assertThat(ex.getMessage(), containsString("Query must be read only. Query: "));
    }

    String FAIL_ANY_CONFIG = formatWithLocale(
        "`%s` and `%s` or `%s` and `%s`",
        NODE_PROJECTION_KEY,
        RELATIONSHIP_PROJECTION_KEY,
        NODE_QUERY_KEY,
        RELATIONSHIP_QUERY_KEY
    );
    // No value specified for the mandatory configuration parameter `relationshipProjection`
    static Stream<Arguments> failingConfigurationMaps() {
        return Stream.of(
            Arguments.of(FAIL_ANY_CONFIG, MapUtil.map()),
            Arguments.of("No value specified for the mandatory configuration parameter `relationshipProjection`", MapUtil.map(NODE_PROJECTION_KEY, ALL_NODES.name)),
            Arguments.of("No value specified for the mandatory configuration parameter `nodeProjection`", MapUtil.map(RELATIONSHIP_PROJECTION_KEY, ALL_RELATIONSHIPS.name)),
            Arguments.of("No value specified for the mandatory configuration parameter `relationshipQuery`", MapUtil.map(NODE_QUERY_KEY, ALL_NODES_QUERY)),
            Arguments.of("No value specified for the mandatory configuration parameter `nodeQuery`", MapUtil.map(RELATIONSHIP_QUERY_KEY, ALL_RELATIONSHIPS_QUERY)),
            Arguments.of(FAIL_ANY_CONFIG, MapUtil.map(NODE_PROJECTION_KEY, ALL_NODES.name, RELATIONSHIP_QUERY_KEY, ALL_RELATIONSHIPS_QUERY)),
            Arguments.of(FAIL_ANY_CONFIG, MapUtil.map(RELATIONSHIP_PROJECTION_KEY, ALL_RELATIONSHIPS.name, NODE_QUERY_KEY, ALL_NODES_QUERY))
        );
    }

    @ParameterizedTest
    @MethodSource("failingConfigurationMaps")
    default void failOnImplicitLoadingWithoutProjectionsOrQueries(String expectedMessage, Map<String, Object> configurationMap) {
        Map<String, Object> config = createMinimalConfig(CypherMapWrapper.create(configurationMap)).toMap();

        applyOnProcedure((proc) -> {
            IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> proc.compute(
                    config,
                    Collections.emptyMap()
                )
            );
            String message = ex.getMessage();
            assertTrue(
                message.contains("Missing information"),
                formatWithLocale("Does not start with 'Missing information': %s", message)
            );
            assertTrue(
                message.contains(expectedMessage),
                formatWithLocale("Does not contain '%s': %s", expectedMessage, message)
            );
        });
    }

    @Test
    default void checkStatsModeExists() {
        applyOnProcedure((proc) -> {
            boolean inStatsClass = methodExists(proc, "stats");
            if (inStatsClass) {
                assertTrue(
                    methodExists(proc, "stats"),
                    formatWithLocale("Expected %s to have a `stats` method", proc.getClass().getSimpleName())
                );
            }
        });
    }

    default boolean methodExists(AlgoBaseProc<?, RESULT, CONFIG> proc, String methodSuffix) {
        return getProcedureMethods(proc)
            .anyMatch(method -> getProcedureMethodName(method).endsWith(methodSuffix));
    }

    default Stream<Method> getProcedureMethods(AlgoBaseProc<?, RESULT, CONFIG> proc) {
        return Arrays.stream(proc.getClass().getDeclaredMethods())
            .filter(method -> method.getDeclaredAnnotation(Procedure.class) != null);
    }

    default String getProcedureMethodName(Method method) {
        Procedure procedureAnnotation = method.getDeclaredAnnotation(Procedure.class);
        Objects.requireNonNull(procedureAnnotation, method + " is not annotation with " + Procedure.class);
        String name = procedureAnnotation.name();
        if (name.isEmpty()) {
            name = procedureAnnotation.value();
        }
        return name;
    }

    default Stream<Method> getWriteAndStreamProcedures(AlgoBaseProc<?, RESULT, CONFIG> proc) {
        return getProcedureMethods(proc)
            .filter(method -> {
                String procedureMethodName = getProcedureMethodName(method);
                return procedureMethodName.endsWith("stream") || procedureMethodName.endsWith("write");
            });
    }

    @NotNull
    default GraphLoader graphLoader(GraphCreateConfig graphCreateConfig) {
        return graphLoader(graphDb(), graphCreateConfig);
    }

    @NotNull
    default GraphLoader graphLoader(
        GraphDatabaseAPI db,
        GraphCreateConfig graphCreateConfig
    ) {
        return ImmutableGraphLoader
            .builder()
            .context(ImmutableGraphLoaderContext.builder()
                .api(db)
                .log(new TestLog())
                .build())
            .username("")
            .createConfig(graphCreateConfig)
            .build();
    }

}
