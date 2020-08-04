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
package org.neo4j.graphalgo.louvain;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.MutateNodePropertyTest;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.StoreLoaderBuilder;
import org.neo4j.graphalgo.api.nodeproperties.ValueType;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.core.Aggregation;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.List;
import java.util.Optional;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.TestSupport.fromGdl;

public class LouvainMutateProcTest extends LouvainProcTest<LouvainMutateConfig> implements MutateNodePropertyTest<Louvain, LouvainMutateConfig, Louvain> {

    @Override
    public String mutateProperty() {
        return "communityId";
    }

    @Override
    public ValueType mutatePropertyType() {
        return ValueType.LONG;
    }

    @Override
    public Optional<String> mutateGraphName() {
        return Optional.of(LOUVAIN_GRAPH);
    }

    @Override
    public String expectedMutatedGraph() {
        return
            "  (a:Node { communityId: 14, seed: 1 })" +
            ", (b:Node { communityId: 14, seed: 1 })" +
            ", (c:Node { communityId: 14, seed: 1 })" +
            ", (d:Node { communityId: 14, seed: 1 })" +
            ", (e:Node { communityId: 14, seed: 1 })" +
            ", (f:Node { communityId: 14, seed: 1 })" +
            ", (g:Node { communityId: 7, seed: 2 })" +
            ", (h:Node { communityId: 7, seed: 2 })" +
            ", (i:Node { communityId: 7, seed: 2 })" +
            ", (j:Node { communityId: 12, seed: 42 })" +
            ", (k:Node { communityId: 12, seed: 42 })" +
            ", (l:Node { communityId: 12, seed: 42 })" +
            ", (m:Node { communityId: 12, seed: 42 })" +
            ", (n:Node { communityId: 12, seed: 42 })" +
            ", (x:Node { communityId: 14, seed: 1 })" +
            // 'LOUVAIN_GRAPH' is UNDIRECTED, e.g. each rel twice
            ", (a)-->(b)-->(a)" +
            ", (a)-->(d)-->(a)" +
            ", (a)-->(f)-->(a)" +
            ", (b)-->(d)-->(b)" +
            ", (b)-->(x)-->(b)" +
            ", (b)-->(g)-->(b)" +
            ", (b)-->(e)-->(b)" +
            ", (c)-->(x)-->(c)" +
            ", (c)-->(f)-->(c)" +
            ", (d)-->(k)-->(d)" +
            ", (e)-->(x)-->(e)" +
            ", (e)-->(f)-->(e)" +
            ", (e)-->(h)-->(e)" +
            ", (f)-->(g)-->(f)" +
            ", (g)-->(h)-->(g)" +
            ", (h)-->(i)-->(h)" +
            ", (h)-->(j)-->(h)" +
            ", (i)-->(k)-->(i)" +
            ", (j)-->(k)-->(j)" +
            ", (j)-->(m)-->(j)" +
            ", (j)-->(n)-->(j)" +
            ", (k)-->(m)-->(k)" +
            ", (k)-->(l)-->(k)" +
            ", (l)-->(n)-->(l)" +
            ", (m)-->(n)-->(m)";
    }

    @Override
    public Class<? extends AlgoBaseProc<Louvain, Louvain, LouvainMutateConfig>> getProcedureClazz() {
        return LouvainMutateProc.class;
    }

    @Override
    public LouvainMutateConfig createConfig(CypherMapWrapper mapWrapper) {
        return LouvainMutateConfig.of(getUsername(), Optional.empty(), Optional.empty(), mapWrapper);
    }

    @Test
    void testMutateAndWriteWithSeeding() {
        var testGraphName = mutateGraphName().get();

        var mutateQuery = GdsCypher
            .call()
            .explicitCreation(testGraphName)
            .algo("louvain")
            .mutateMode()
            .addParameter("mutateProperty", mutateProperty())
            .yields();

        runQuery(mutateQuery);

        var writeQuery = GdsCypher
            .call()
            .explicitCreation(testGraphName)
            .algo("louvain")
            .writeMode()
            .addParameter("seedProperty", mutateProperty())
            .addParameter("writeProperty", mutateProperty())
            .yields();

        runQuery(writeQuery);

        var updatedGraph = new StoreLoaderBuilder().api(db)
            .addNodeLabel("Node")
            .globalOrientation(Orientation.UNDIRECTED)
            .addNodeProperty(mutateProperty(), mutateProperty(), 42.0, Aggregation.NONE)
            .addNodeProperty("seed", "seed", 42.0, Aggregation.NONE)
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
            .algo("louvain")
            .mutateMode()
            .addParameter("mutateProperty", mutateProperty())
            .yields(
                "nodePropertiesWritten",
                "createMillis",
                "computeMillis",
                "mutateMillis",
                "postProcessingMillis",
                "ranLevels",
                "communityCount",
                "modularities",
                "communityDistribution",
                "configuration"
            );

        runQueryWithRowConsumer(
            query,
            row -> {
                assertEquals(15L, row.getNumber("nodePropertiesWritten"));

                assertThat(-1L, lessThan(row.getNumber("createMillis").longValue()));
                assertThat(-1L, lessThan(row.getNumber("computeMillis").longValue()));
                assertThat(-1L, lessThan(row.getNumber("mutateMillis").longValue()));

                assertEquals(1L, row.get("ranLevels"));
                assertEquals(4L, row.getNumber("communityCount"));
                assertEquals(0.3744, ((List<Double>) row.get("modularities")).get(0), 1E-3);

                assertEquals(MapUtil.map(
                    "p99", 8L,
                    "min", 2L,
                    "max", 8L,
                    "mean", 3.75D,
                    "p90", 8L,
                    "p50", 2L,
                    "p999", 8L,
                    "p95", 8L,
                    "p75", 3L
                ), row.get("communityDistribution"));
            }
        );
    }

}
