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
package org.neo4j.graphalgo.labelpropagation;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.MutateNodePropertyTest;
import org.neo4j.graphalgo.StoreLoaderBuilder;
import org.neo4j.graphalgo.api.DefaultValue;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.TestSupport.fromGdl;

public class LabelPropagationMutateProcTest extends LabelPropagationProcTest<LabelPropagationMutateConfig> implements MutateNodePropertyTest<LabelPropagation, LabelPropagationMutateConfig, LabelPropagation> {

    @Override
    public String mutateProperty() {
        return "communityId";
    }

    @Override
    public ValueType mutatePropertyType() {
        return ValueType.LONG;
    }

    @Override
    public String expectedMutatedGraph() {
        return
            "  (a { communityId: 2 }) " +
            ", (b { communityId: 7 }) " +
            ", (a)-->({ communityId: 2 }) " +
            ", (a)-->({ communityId: 3 }) " +
            ", (a)-->({ communityId: 4 }) " +
            ", (a)-->({ communityId: 5 }) " +
            ", (a)-->({ communityId: 6 }) " +
            ", (b)-->({ communityId: 7 }) " +
            ", (b)-->({ communityId: 8 }) " +
            ", (b)-->({ communityId: 9 }) " +
            ", (b)-->({ communityId: 10 }) " +
            ", (b)-->({ communityId: 11 })";
    }

    @Override
    public Class<? extends AlgoBaseProc<LabelPropagation, LabelPropagation, LabelPropagationMutateConfig>> getProcedureClazz() {
        return LabelPropagationMutateProc.class;
    }

    @Override
    public LabelPropagationMutateConfig createConfig(CypherMapWrapper mapWrapper) {
        return LabelPropagationMutateConfig.of(getUsername(), Optional.empty(), Optional.empty(), mapWrapper);
    }

    @Test
    void testMutateAndWriteWithSeeding() {
        var testGraphName = "lpaGraph";
        var initialGraphStore = new StoreLoaderBuilder().api(db)
            .build()
            .graphStore();

        GraphStoreCatalog.set(emptyWithNameNative(getUsername(), testGraphName), initialGraphStore);

        var mutateQuery = GdsCypher
            .call()
            .explicitCreation(testGraphName)
            .algo("labelPropagation")
            .mutateMode()
            .addParameter("mutateProperty", mutateProperty())
            .yields();

        runQuery(mutateQuery);

        var writeQuery = GdsCypher
            .call()
            .explicitCreation(testGraphName)
            .algo("labelPropagation")
            .writeMode()
            .addParameter("seedProperty", mutateProperty())
            .addParameter("writeProperty", mutateProperty())
            .yields();

        runQuery(writeQuery);

        var updatedGraph = new StoreLoaderBuilder().api(db)
            .addNodeProperty(mutateProperty(), mutateProperty(), DefaultValue.of(42.0), Aggregation.NONE)
            .build()
            .graph();

        assertGraphEquals(fromGdl(expectedMutatedGraph()), updatedGraph);
    }

    @Test
    void testMutateYields() {
        String query = GdsCypher
            .call()
            .withAnyLabel()
            .withAnyRelationshipType()
            .algo("labelPropagation")
            .mutateMode()
            .addParameter("mutateProperty", mutateProperty())
            .yields();

        runQueryWithRowConsumer(
            query,
            row -> {
                assertThat(-1L, lessThan(row.getNumber("createMillis").longValue()));
                assertThat(-1L, lessThan(row.getNumber("computeMillis").longValue()));
                assertThat(-1L, lessThan(row.getNumber("mutateMillis").longValue()));
                assertThat(-1L, lessThan(row.getNumber("postProcessingMillis").longValue()));

                assertEquals(12L, row.getNumber("nodePropertiesWritten").longValue());
                assertEquals(10L, row.getNumber("communityCount"));
                assertTrue(row.getBoolean("didConverge"));

                assertEquals(MapUtil.map(
                    "p99", 2L,
                    "min", 1L,
                    "max", 2L,
                    "mean", 1.2D,
                    "p90", 2L,
                    "p50", 1L,
                    "p999", 2L,
                    "p95", 2L,
                    "p75", 1L
                ), row.get("communityDistribution"));
            }
        );
    }

    @Test
    void testGraphMutationFiltered() {
        long deletedNodes = clearDb();
        runQuery("CREATE (x:Ignore {id: -1, communityId: null}) " + createQuery());

        String graphName = "loadGraph";

        String loadQuery = GdsCypher
            .call()
            .withNodeLabels("Ignore", "A", "B")
            .withAnyRelationshipType()
            .graphCreate(graphName)
            .yields();

        runQuery(loadQuery);

        String query = GdsCypher
            .call()
            .explicitCreation(graphName)
            .algo("labelPropagation")
            .mutateMode()
            .addParameter("nodeLabels", Arrays.asList("A", "B"))
            .addParameter("mutateProperty", mutateProperty())
            .yields();

        runQuery(query);

        List<Long> expectedValueList = new ArrayList<>(RESULT.size() + 1);
        expectedValueList.add(Long.MIN_VALUE);
        RESULT.forEach(component -> expectedValueList.add(component + deletedNodes + 1));

        Graph mutatedGraph = GraphStoreCatalog.get(TEST_USERNAME, namedDatabaseId(), graphName).graphStore().getUnion();
        mutatedGraph.forEachNode(nodeId -> {
            assertEquals(
                    expectedValueList.get(Math.toIntExact(nodeId)),
                    mutatedGraph.nodeProperties("communityId").longValue(nodeId)
                );
                return true;
            }
        );
    }
}
