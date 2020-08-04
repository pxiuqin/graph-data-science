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
package org.neo4j.gds.embeddings.graphsage.proc;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.provider.Arguments;
import org.neo4j.gds.embeddings.graphsage.ActivationFunction;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.Orientation;
import org.neo4j.graphalgo.PropertyMapping;
import org.neo4j.graphalgo.RelationshipProjection;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;

import java.util.stream.Stream;

import static org.neo4j.graphalgo.TestSupport.crossArguments;

class GraphSageBaseProcTest extends BaseProcTest {

    private static final String DB_CYPHER =
        "CREATE" +
        "  (a:King{ name: 'A', age: 20, birth_year: 200, death_year: 300 })" +
        ", (b:King{ name: 'B', age: 12, birth_year: 232, death_year: 300 })" +
        ", (c:King{ name: 'C', age: 67, birth_year: 212, death_year: 300 })" +
        ", (d:King{ name: 'D', age: 78, birth_year: 245, death_year: 300 })" +
        ", (e:King{ name: 'E', age: 32, birth_year: 256, death_year: 300 })" +
        ", (f:King{ name: 'F', age: 32, birth_year: 214, death_year: 300 })" +
        ", (g:King{ name: 'G', age: 35, birth_year: 214, death_year: 300 })" +
        ", (h:King{ name: 'H', age: 56, birth_year: 253, death_year: 300 })" +
        ", (i:King{ name: 'I', age: 62, birth_year: 267, death_year: 300 })" +
        ", (j:King{ name: 'J', age: 44, birth_year: 289, death_year: 300 })" +
        ", (k:King{ name: 'K', age: 89, birth_year: 211, death_year: 300 })" +
        ", (l:King{ name: 'L', age: 99, birth_year: 201, death_year: 300 })" +
        ", (m:King{ name: 'M', age: 99, birth_year: 201, death_year: 300 })" +
        ", (n:King{ name: 'N', age: 99, birth_year: 201, death_year: 300 })" +
        ", (o:King{ name: 'O', age: 99, birth_year: 201, death_year: 300 })" +
        ", (a)-[:REL]->(b)" +
        ", (a)-[:REL]->(c)" +
        ", (b)-[:REL]->(c)" +
        ", (b)-[:REL]->(d)" +
        ", (c)-[:REL]->(e)" +
        ", (d)-[:REL]->(e)" +
        ", (d)-[:REL]->(f)" +
        ", (e)-[:REL]->(f)" +
        ", (e)-[:REL]->(g)" +
        ", (h)-[:REL]->(i)" +
        ", (i)-[:REL]->(j)" +
        ", (j)-[:REL]->(k)" +
        ", (j)-[:REL]->(l)" +
        ", (k)-[:REL]->(l)";


    @BeforeEach
    void setup() throws Exception {
        registerProcedures(
            GraphCreateProc.class,
            GraphSageStreamProc.class,
            GraphSageWriteProc.class
        );

        runQuery(DB_CYPHER);

        String query = GdsCypher.call()
            .withNodeLabel("King")
            .withNodeProperty(PropertyMapping.of("age", 1.0))
            .withNodeProperty(PropertyMapping.of("birth_year", 1.0))
            .withNodeProperty(PropertyMapping.of("death_year", 1.0))
            .withRelationshipType(
                "R",
                RelationshipProjection.of(
                    "*",
                    Orientation.UNDIRECTED
                )
            )
            .graphCreate("embeddingsGraph")
            .yields();

        runQuery(query);
    }

    @AfterEach
    void tearDown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }


    protected static Stream<Arguments> configVariations() {
        return crossArguments(
            () -> Stream.of(
                Arguments.of(16),
                Arguments.of(32),
                Arguments.of(64)
            ),
            () -> Stream.of(
                Arguments.of("mean"),
                Arguments.of("pool")
            ),
            () -> Stream.of(
                Arguments.of(ActivationFunction.SIGMOID),
                Arguments.of(ActivationFunction.RELU)
            )
        );
    }
}
