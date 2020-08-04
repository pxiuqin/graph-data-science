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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;

import java.util.Map;
import java.util.Optional;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.graphalgo.Orientation.REVERSE;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.TestSupport.fromGdl;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public class NodeSimilarityWriteProcTest extends NodeSimilarityProcTest<NodeSimilarityWriteConfig> {

    @Override
    public Class<? extends AlgoBaseProc<NodeSimilarity, NodeSimilarityResult, NodeSimilarityWriteConfig>> getProcedureClazz() {
        return NodeSimilarityWriteProc.class;
    }

    @Override
    public NodeSimilarityWriteConfig createConfig(CypherMapWrapper mapWrapper) {
        return NodeSimilarityWriteConfig.of("", Optional.empty(), Optional.empty(), mapWrapper);
    }

    @ParameterizedTest(name = "{2}")
    @MethodSource("org.neo4j.graphalgo.nodesim.NodeSimilarityProcTest#allValidGraphVariationsWithProjections")
    void shouldWriteResults(GdsCypher.QueryBuilder queryBuilder, Orientation orientation, String testName) {
        String query = queryBuilder
            .algo("nodeSimilarity")
            .writeMode()
            .addParameter("similarityCutoff", 0.0)
            .addParameter("writeRelationshipType", "SIMILAR")
            .addParameter("writeProperty", "score")
            .yields(
                "computeMillis",
                "createMillis",
                "nodesCompared ",
                "relationshipsWritten",
                "writeMillis",
                "similarityDistribution",
                "postProcessingMillis",
                "configuration"
            );

        runQueryWithRowConsumer(query, row -> {
            assertEquals(3, row.getNumber("nodesCompared").longValue());
            assertEquals(6, row.getNumber("relationshipsWritten").longValue());
            assertUserInput(row, "writeRelationshipType", "SIMILAR");
            assertUserInput(row, "writeProperty", "score");
            assertThat("Missing computeMillis", -1L, lessThan(row.getNumber("computeMillis").longValue()));
            assertThat("Missing createMillis", -1L, lessThan(row.getNumber("createMillis").longValue()));
            assertThat("Missing writeMillis", -1L, lessThan(row.getNumber("writeMillis").longValue()));

            Map<String, Double> distribution = (Map<String, Double>) row.get("similarityDistribution");
            assertThat("Missing min", -1.0, lessThan(distribution.get("min")));
            assertThat("Missing max", -1.0, lessThan(distribution.get("max")));
            assertThat("Missing mean", -1.0, lessThan(distribution.get("mean")));
            assertThat("Missing stdDev", -1.0, lessThan(distribution.get("stdDev")));
            assertThat("Missing p1", -1.0, lessThan(distribution.get("p1")));
            assertThat("Missing p5", -1.0, lessThan(distribution.get("p5")));
            assertThat("Missing p10", -1.0, lessThan(distribution.get("p10")));
            assertThat("Missing p25", -1.0, lessThan(distribution.get("p25")));
            assertThat("Missing p50", -1.0, lessThan(distribution.get("p50")));
            assertThat("Missing p75", -1.0, lessThan(distribution.get("p75")));
            assertThat("Missing p90", -1.0, lessThan(distribution.get("p90")));
            assertThat("Missing p95", -1.0, lessThan(distribution.get("p95")));
            assertThat("Missing p99", -1.0, lessThan(distribution.get("p99")));
            assertThat("Missing p100", -1.0, lessThan(distribution.get("p100")));

            assertThat(
                "Missing postProcessingMillis",
                -1L,
                equalTo(row.getNumber("postProcessingMillis").longValue())
            );
        });

        String resultGraphName = "simGraph_" + orientation.name();
        String loadQuery = GdsCypher.call()
            .withNodeLabel(orientation == REVERSE ? "Item" : "Person")
            .withRelationshipType("SIMILAR", orientation)
            .withNodeProperty("id")
            .withRelationshipProperty("score")
            .graphCreate(resultGraphName)
            .yields();

        runQuery(loadQuery);

        Graph simGraph = GraphStoreCatalog.getUnion(getUsername(), namedDatabaseId(), resultGraphName).orElse(null);
        assertNotNull(simGraph);
        assertGraphEquals(
            orientation == REVERSE
                ? fromGdl(
                formatWithLocale(
                    "  (i1:Item {id: 10})" +
                    ", (i2:Item {id: 11})" +
                    ", (i3:Item {id: 12})" +
                    ", (i4:Item {id: 13})" +
                    ", (i1)-[{w: %f}]->(i2)" +
                    ", (i1)-[{w: %f}]->(i3)" +
                    ", (i2)-[{w: %f}]->(i1)" +
                    ", (i2)-[{w: %f}]->(i3)" +
                    ", (i3)-[{w: %f}]->(i1)" +
                    ", (i3)-[{w: %f}]->(i2)",
                    1 / 1.0,
                    1 / 3.0,
                    1 / 1.0,
                    1 / 3.0,
                    1 / 3.0,
                    1 / 3.0
                )
            )
                : fromGdl(
                    formatWithLocale(
                        "  (a:Person {id: 0})" +
                        ", (b:Person {id: 1})" +
                        ", (c:Person {id: 2})" +
                        ", (d:Person {id: 3})" +
                        ", (a)-[{w: %f}]->(b)" +
                        ", (a)-[{w: %f}]->(c)" +
                        ", (b)-[{w: %f}]->(c)" +
                        ", (b)-[{w: %f}]->(a)" +
                        ", (c)-[{w: %f}]->(a)" +
                        ", (c)-[{w: %f}]->(b)"
                        , 2 / 3.0
                        , 1 / 3.0
                        , 0.0
                        , 2 / 3.0
                        , 1 / 3.0
                        , 0.0
                    )
                ),
            simGraph
        );
    }

    @ParameterizedTest(name = "missing parameter: {0}")
    @ValueSource(strings = {"writeProperty", "writeRelationshipType"})
    void shouldFailIfConfigIsMissingWriteParameters(String parameter) {
        CypherMapWrapper input = createMinimalConfig(CypherMapWrapper.empty())
            .withoutEntry(parameter);

        IllegalArgumentException illegalArgumentException = assertThrows(
            IllegalArgumentException.class,
            () -> createConfig(input)
        );
        assertThat(
            illegalArgumentException.getMessage(),
            startsWith(formatWithLocale("No value specified for the mandatory configuration parameter `%s`", parameter))
        );
    }

    @Override
    public CypherMapWrapper createMinimalConfig(CypherMapWrapper mapWrapper) {
        if (!mapWrapper.containsKey("writeProperty")) {
            mapWrapper = mapWrapper.withString("writeProperty", "foo");
        }
        if (!mapWrapper.containsKey("writeRelationshipType")) {
            mapWrapper = mapWrapper.withString("writeRelationshipType", "bar");
        }
        return mapWrapper;
    }
}
