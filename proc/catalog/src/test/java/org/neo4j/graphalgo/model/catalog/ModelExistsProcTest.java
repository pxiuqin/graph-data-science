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
package org.neo4j.graphalgo.model.catalog;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.core.model.Model;
import org.neo4j.graphalgo.core.model.ModelCatalog;

import java.util.Map;

import static java.util.Collections.singletonList;
import static org.neo4j.graphalgo.compat.MapUtil.map;

class ModelExistsProcTest  extends BaseProcTest {

    private static final String EXISTS_QUERY = "CALL gds.beta.model.exists($modelName)";

    @BeforeEach
    void setUp() throws Exception {
        registerProcedures(ModelExistsProc.class);
    }

    @Test
    void checksIfModelExists() {
        String existingModel = "testModel";
        ModelCatalog.set(Model.of(getUsername(), existingModel, "testAlgo", "testData", TestTrainConfig.of()));

        assertCypherResult(
            EXISTS_QUERY,
            Map.of(
                "modelName", existingModel
            ),
            singletonList(
                map(
                    "modelName", existingModel,
                    "modelType", "testAlgo",
                    "exists", true
                )
            )
        );
    }

    @Test
    void returnsCorrectResultForNonExistingModel() {

        String bogusModel = "bogusModel";
        assertCypherResult(
            EXISTS_QUERY,
            Map.of(
                "modelName", bogusModel),
            singletonList(
                map(
                    "modelName", bogusModel,
                    "modelType", "n/a",
                    "exists", false
                )
            )
        );
    }
}
