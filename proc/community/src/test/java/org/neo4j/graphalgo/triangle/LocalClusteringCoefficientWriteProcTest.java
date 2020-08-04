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
package org.neo4j.graphalgo.triangle;

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.WritePropertyConfigTest;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.isA;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

class LocalClusteringCoefficientWriteProcTest
    extends LocalClusteringCoefficientBaseProcTest<LocalClusteringCoefficientWriteConfig>
    implements WritePropertyConfigTest<LocalClusteringCoefficient, LocalClusteringCoefficientWriteConfig, LocalClusteringCoefficient.Result> {

    @Test
    void testWrite() {
        var query = "CALL gds.localClusteringCoefficient.write('g', { writeProperty: 'localCC' })";

        assertCypherResult(query, List.of(Map.of(
            "averageClusteringCoefficient", closeTo(expectedAverageClusteringCoefficient() / 5, 1e-10),
            "nodeCount", 5L,
            "createMillis", greaterThan(-1L),
            "computeMillis", greaterThan(-1L),
            "configuration", isA(Map.class),
            "nodePropertiesWritten", 5L,
            "writeMillis", greaterThan(-1L)
        )));

        assertWriteResult(expectedResult, "localCC");
    }

    @Test
    void testWriteSeeded() {
        var query = "CALL gds.localClusteringCoefficient.write('g', { " +
                    "   writeProperty: 'localCC', " +
                    "   triangleCountProperty: 'seed' " +
                    "})";

        assertCypherResult(query, List.of(Map.of(
            "averageClusteringCoefficient", closeTo(expectedAverageClusteringCoefficientSeeded() / 5, 1e-10),
            "nodeCount", 5L,
            "createMillis", greaterThan(-1L),
            "computeMillis", greaterThan(-1L),
            "configuration", isA(Map.class),
            "nodePropertiesWritten", 5L,
            "writeMillis", greaterThan(-1L)
        )));

        assertWriteResult(expectedResultWithSeeding, "localCC");
    }

    @Override
    public Class<? extends AlgoBaseProc<LocalClusteringCoefficient, LocalClusteringCoefficient.Result, LocalClusteringCoefficientWriteConfig>> getProcedureClazz() {
        return LocalClusteringCoefficientWriteProc.class;
    }

    @Override
    public LocalClusteringCoefficientWriteConfig createConfig(CypherMapWrapper mapWrapper) {
        return LocalClusteringCoefficientWriteConfig.of(
            getUsername(),
            Optional.empty(),
            Optional.empty(),
            mapWrapper
        );
    }

    @Override
    public CypherMapWrapper createMinimalConfig(CypherMapWrapper mapWrapper) {
        if (!mapWrapper.containsKey("writeProperty")) {
            mapWrapper = mapWrapper.withString("writeProperty", "writeProperty");
        }
        return mapWrapper;
    }

    private void assertWriteResult(
        Map<String, Double> expectedResult,
        String writeProperty
    ) {
        runQueryWithRowConsumer(String.format(
            "MATCH (n) RETURN n.name AS name, n.%s AS localCC",
            writeProperty
        ), (row) -> {
            double lcc = row.getNumber("localCC").doubleValue();
            String name = row.getString("name");
            Double expectedLcc = expectedResult.get(name);
            assertEquals(expectedLcc, lcc, formatWithLocale("Node with name `%s` has wrong coefficient", name));
        });
    }

}
