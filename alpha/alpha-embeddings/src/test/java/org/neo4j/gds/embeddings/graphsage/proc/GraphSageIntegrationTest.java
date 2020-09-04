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

import org.junit.jupiter.api.Test;
import org.neo4j.gds.embeddings.graphsage.algo.GraphSage;
import org.neo4j.graphalgo.GdsCypher;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonList;
import static org.hamcrest.Matchers.aMapWithSize;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.hasEntry;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.gds.embeddings.graphsage.ActivationFunction.SIGMOID;

class GraphSageIntegrationTest extends GraphSageBaseProcTest {

    private static final int EMBEDDING_SIZE = 42;

    @Test
    void shouldRun() {
        modelDoesntExist();

        train(EMBEDDING_SIZE, "mean", SIGMOID);

        modelExists();

        stream();

        dropModel();

        modelDoesntExist();
    }

    private void modelDoesntExist() {
        checkModelExistence("n/a", false);
    }

    private void modelExists() {
        checkModelExistence(GraphSage.MODEL_TYPE, true);
    }

    private void checkModelExistence(String modelType, boolean exists) {
        assertCypherResult(
            "CALL gds.beta.model.exists($modelName)",
            Map.of("modelName", modelName),
            List.of(
                Map.of(
                    "modelName", modelName,
                    "modelType", modelType,
                    "exists", exists
                )
            )
        );
    }

    private void stream() {
        String streamQuery = GdsCypher.call().explicitCreation("embeddingsGraph")
            .algo("gds.alpha.graphSage")
            .streamMode()
            .addParameter("concurrency", 1)
            .addParameter("modelName", modelName)
            .yields();

        runQueryWithRowConsumer(streamQuery, row -> {
            Number nodeId = row.getNumber("nodeId");
            assertNotNull(nodeId);

            Object o = row.get("embedding");
            assertTrue(o instanceof List);
            Collection<Double> nodeEmbeddings = (List<Double>) o;
            assertEquals(GraphSageIntegrationTest.EMBEDDING_SIZE, nodeEmbeddings.size());
        });
    }

    private void dropModel() {
        assertCypherResult(
            "CALL gds.beta.model.drop($modelName)",
            Map.of("modelName", modelName),
            singletonList(
                Map.of(
                    "modelInfo", Map.of("modelName", modelName, "modelType", GraphSage.MODEL_TYPE),
                    "trainConfig", allOf(
                        aMapWithSize(18),
                        hasEntry("modelName", modelName),
                        hasEntry("aggregator", "MEAN"),
                        hasEntry("activationFunction", "SIGMOID")
                    )
                )
            )
        );
    }
}
