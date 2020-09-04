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
package org.neo4j.graphalgo.betweenness;

import org.junit.jupiter.api.BeforeEach;
import org.neo4j.graphalgo.AlgoBaseProcTest;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.MemoryEstimateTest;
import org.neo4j.graphalgo.OrientationCombinationTest;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.catalog.GraphWriteNodePropertiesProc;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicDoubleArray;
import org.neo4j.graphdb.Label;
import org.neo4j.kernel.internal.GraphDatabaseAPI;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.graphalgo.compat.GraphDatabaseApiProxy.runInTransaction;

abstract class BetweennessCentralityProcTest<CONFIG extends BetweennessCentralityBaseConfig>
    extends BaseProcTest
    implements
    AlgoBaseProcTest<BetweennessCentrality, CONFIG, HugeAtomicDoubleArray>,
    MemoryEstimateTest<BetweennessCentrality, CONFIG, HugeAtomicDoubleArray>,
    OrientationCombinationTest<BetweennessCentrality, CONFIG, HugeAtomicDoubleArray> {

    static final String DEFAULT_RESULT_PROPERTY = "centrality";

    static Map<Long, Double> EXPECTED = new HashMap<>();

    @Override
    public String createQuery() {
        return "CREATE" +
               "  (a:Node {name: 'a'})" +
               ", (b:Node {name: 'b'})" +
               ", (c:Node {name: 'c'})" +
               ", (d:Node {name: 'd'})" +
               ", (e:Node {name: 'e'})" +
               ", (a)-[:REL]->(b)" +
               ", (b)-[:REL]->(c)" +
               ", (c)-[:REL]->(d)" +
               ", (d)-[:REL]->(e)";
    }

    @Override
    public GraphDatabaseAPI graphDb() {
        return db;
    }

    @BeforeEach
    void setupGraph() throws Exception {
        registerProcedures(
            BetweennessCentralityStreamProc.class,
            BetweennessCentralityWriteProc.class,
            BetweennessCentralityMutateProc.class,
            BetweennessCentralityStatsProc.class,
            GraphWriteNodePropertiesProc.class,
            GraphCreateProc.class
        );

        runQuery(createQuery());

        runInTransaction(db, tx -> {
            var label = Label.label("Node");
            EXPECTED.put(tx.findNode(label, "name", "a").getId(), 0.0);
            EXPECTED.put(tx.findNode(label, "name", "b").getId(), 3.0);
            EXPECTED.put(tx.findNode(label, "name", "c").getId(), 4.0);
            EXPECTED.put(tx.findNode(label, "name", "d").getId(), 3.0);
            EXPECTED.put(tx.findNode(label, "name", "e").getId(), 0.0);
        });
    }

    @Override
    public void assertResultEquals(HugeAtomicDoubleArray result1, HugeAtomicDoubleArray result2) {
        assertEquals(result1.size(), result2.size());

        for (long nodeId = 0; nodeId < result1.size(); nodeId++) {
            assertEquals(result1.get(nodeId), result2.get(nodeId));
        }
    }

}
