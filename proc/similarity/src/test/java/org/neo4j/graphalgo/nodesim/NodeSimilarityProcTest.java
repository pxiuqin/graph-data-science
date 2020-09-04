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
package org.neo4j.graphalgo.nodesim;

import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.utility.Iterate;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.neo4j.graphalgo.AlgoBaseProcTest;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.HeapControlTest;
import org.neo4j.graphalgo.MemoryEstimateTest;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.RelationshipWeightConfigTest;
import org.neo4j.graphalgo.TestSupport;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.catalog.GraphWriteNodePropertiesProc;
import org.neo4j.graphalgo.catalog.GraphWriteRelationshipProc;
import org.neo4j.graphalgo.config.ConcurrencyConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.Collection;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.params.provider.Arguments.arguments;
import static org.neo4j.graphalgo.Orientation.NATURAL;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

abstract class NodeSimilarityProcTest<CONFIG extends NodeSimilarityBaseConfig> extends BaseProcTest implements
    AlgoBaseProcTest<NodeSimilarity, CONFIG, NodeSimilarityResult>,
    MemoryEstimateTest<NodeSimilarity, CONFIG, NodeSimilarityResult>,
    HeapControlTest<NodeSimilarity, CONFIG, NodeSimilarityResult>,
    RelationshipWeightConfigTest<NodeSimilarity, CONFIG, NodeSimilarityResult> {

    @Override
    public String createQuery() {
        return "CREATE" +
               "  (a:Person {id: 0,  name: 'Alice'})" +
               ", (b:Person {id: 1,  name: 'Bob'})" +
               ", (c:Person {id: 2,  name: 'Charlie'})" +
               ", (d:Person {id: 3,  name: 'Dave'})" +
               ", (i1:Item  {id: 10, name: 'p1'})" +
               ", (i2:Item  {id: 11, name: 'p2'})" +
               ", (i3:Item  {id: 12, name: 'p3'})" +
               ", (i4:Item  {id: 13, name: 'p4'})" +
               ", (a)-[:LIKES]->(i1)" +
               ", (a)-[:LIKES]->(i2)" +
               ", (a)-[:LIKES]->(i3)" +
               ", (b)-[:LIKES]->(i1)" +
               ", (b)-[:LIKES]->(i2)" +
               ", (c)-[:LIKES]->(i3)";
    }

    @BeforeEach
    void setup() throws Exception {
        registerProcedures(
            NodeSimilarityWriteProc.class,
            NodeSimilarityStreamProc.class,
            NodeSimilarityStatsProc.class,
            NodeSimilarityMutateProc.class,
            GraphCreateProc.class,
            GraphWriteNodePropertiesProc.class,
            GraphWriteRelationshipProc.class
        );
        runQuery(createQuery());

        TestSupport.allDirectedProjections().forEach(orientation -> {
            String name = "myGraph" + orientation.name();
            String createQuery = GdsCypher.call()
                .withAnyLabel()
                .withRelationshipType(
                    "LIKES",
                    RelationshipProjection.builder().type("LIKES").orientation(orientation).build()
                )
                .graphCreate(name)
                .yields();
            runQuery(createQuery);
        });
    }

    @AfterEach
    void teardown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    static Stream<Arguments> allGraphVariations() {
        return graphVariationForProjection(NATURAL).map(args -> arguments(args.get()[0], args.get()[2]));
    }

    static Stream<Arguments> allValidGraphVariationsWithProjections() {
        return TestSupport.allDirectedProjections().flatMap(NodeSimilarityProcTest::graphVariationForProjection);
    }

    private static Stream<Arguments> graphVariationForProjection(Orientation orientation) {
        String name = "myGraph" + orientation.name();
        return Stream.of(
            arguments(
                GdsCypher.call().explicitCreation(name),
                orientation,
                "explicit graph - " + orientation
            ),
            arguments(
                GdsCypher
                    .call()
                    .withNodeLabels("Person", "Item")
                    .withRelationshipType("LIKES", RelationshipProjection
                        .builder()
                        .type("LIKES")
                        .orientation(orientation)
                        .build()
                    ),
                orientation,
                "implicit graph - " + orientation
            )
        );
    }

    @Override
    public GraphDatabaseAPI graphDb() {
        return db;
    }

    @Override
    public void assertResultEquals(NodeSimilarityResult result1, NodeSimilarityResult result2) {
        Optional<Stream<SimilarityResult>> maybeStream1 = result1.maybeStreamResult();
        if (maybeStream1.isPresent()) {
            Optional<Stream<SimilarityResult>> maybeStream2 = result2.maybeStreamResult();
            assertTrue(
                maybeStream2.isPresent(),
                "The two results are of different kind, left is a stream result, right is a graph result."
            );
            Collection<Pair<SimilarityResult, SimilarityResult>> comparableResults = Iterate.zip(
                maybeStream1.get().collect(Collectors.toList()),
                maybeStream2.get().collect(Collectors.toList())
            );
            for (Pair<SimilarityResult, SimilarityResult> pair : comparableResults) {
                SimilarityResult left = pair.getOne();
                SimilarityResult right = pair.getTwo();
                assertEquals(left, right);
            }
            return;
        }
        Optional<SimilarityGraphResult> maybeGraph1 = result1.maybeGraphResult();
        if (maybeGraph1.isPresent()) {
            Optional<SimilarityGraphResult> maybeGraph2 = result2.maybeGraphResult();
            assertTrue(
                maybeGraph2.isPresent(),
                "The two results are of different kind, left is a graph result, right is a stream result."
            );
            assertGraphEquals(maybeGraph1.get().similarityGraph(), maybeGraph2.get().similarityGraph());
            return;
        }

        fail("Result is neither a stream result or a graph result. Congratulations, this should never happen.");
    }

    @ParameterizedTest(name = "parameter: {0}, value: {1}")
    @CsvSource(value = {"topN, -2", "bottomN, -2", "topK, -2", "bottomK, -2", "topK, 0", "bottomK, 0"})
    void shouldThrowForInvalidTopsAndBottoms(String parameter, long value) {
        String message = formatWithLocale("Value for `%s` must be within", parameter);
        CypherMapWrapper input = baseUserInput().withNumber(parameter, value);

        IllegalArgumentException illegalArgumentException = assertThrows(
            IllegalArgumentException.class,
            () -> config(input)
        );
        assertThat(illegalArgumentException.getMessage(), containsString(message));
    }

    @ParameterizedTest
    @CsvSource(value = {"topK, bottomK", "topN, bottomN"})
    void shouldThrowForInvalidTopAndBottomCombination(String top, String bottom) {
        CypherMapWrapper input = baseUserInput().withNumber(top, 1).withNumber(bottom, 1);

        String expectedMessage = formatWithLocale("Invalid parameter combination: %s combined with %s", top, bottom);

        IllegalArgumentException illegalArgumentException = assertThrows(
            IllegalArgumentException.class,
            () -> config(input)
        );
        assertThat(illegalArgumentException.getMessage(), is(expectedMessage));
    }

    @Test
    void shouldThrowIfDegreeCutoffSetToZero() {
        CypherMapWrapper input = baseUserInput().withNumber("degreeCutoff", 0);

        IllegalArgumentException illegalArgumentException = assertThrows(
            IllegalArgumentException.class,
            () -> config(input)
        );
        assertThat(illegalArgumentException.getMessage(), is(formatWithLocale("Value for `degreeCutoff` must be within [1, %d].", Integer.MAX_VALUE)));
    }

    @ParameterizedTest
    @ValueSource(doubles = {-4.2, 4.2})
    void shouldThrowIfSimilarityCutoffIsOutOfRange(double cutoff) {
        CypherMapWrapper input = baseUserInput().withNumber("similarityCutoff", cutoff);

        IllegalArgumentException illegalArgumentException = assertThrows(
            IllegalArgumentException.class,
            () -> config(input)
        );
        assertThat(
            illegalArgumentException.getMessage(),
            is(formatWithLocale("Value for `similarityCutoff` must be within [%.2f, %.2f].", 0D, 1D))
        );
    }

    @Test
    void shouldCreateValidDefaultAlgoConfig() {
        CypherMapWrapper input = baseUserInput();
        NodeSimilarityBaseConfig config = config(input);

        assertEquals(10, config.normalizedK());
        assertEquals(0, config.normalizedN());
        assertEquals(1, config.degreeCutoff());
        assertEquals(1E-42, config.similarityCutoff());
        assertEquals(ConcurrencyConfig.DEFAULT_CONCURRENCY, config.concurrency());
    }

    @ParameterizedTest(name = "top or bottom: {0}")
    @ValueSource(strings = {"top", "bottom"})
    void shouldCreateValidCustomAlgoConfig(String parameter) {
        CypherMapWrapper input = baseUserInput()
            .withNumber(parameter + "K", 100)
            .withNumber(parameter + "N", 1000)
            .withNumber("degreeCutoff", 42)
            .withNumber("similarityCutoff", 0.23)
            .withNumber("concurrency", 1);

        NodeSimilarityBaseConfig config = config(input);

        assertEquals(parameter.equals("top") ? 100 : -100, config.normalizedK());
        assertEquals(parameter.equals("top") ? 1000 : -1000, config.normalizedN());
        assertEquals(42, config.degreeCutoff());
        assertEquals(0.23, config.similarityCutoff());
        assertEquals(1, config.concurrency());
    }

    private CypherMapWrapper baseUserInput() {
        return createMinimalConfig(CypherMapWrapper.empty());
    }

    private CONFIG config(CypherMapWrapper input) {
        return createConfig(input);
    }
}
