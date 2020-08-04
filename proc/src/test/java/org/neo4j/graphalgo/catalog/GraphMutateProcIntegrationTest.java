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
package org.neo4j.graphalgo.catalog;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.gds.embeddings.graphsage.ddl4j.Tensor;
import org.neo4j.gds.embeddings.graphsage.ddl4j.functions.L2Norm;
import org.neo4j.gds.embeddings.graphsage.proc.GraphSageStreamProc;
import org.neo4j.graphalgo.BaseProcTest;
import org.neo4j.graphalgo.GdsCypher;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.labelpropagation.LabelPropagationMutateProc;
import org.neo4j.graphalgo.louvain.LouvainMutateProc;
import org.neo4j.graphalgo.nodesim.NodeSimilarityMutateProc;
import org.neo4j.graphalgo.pagerank.PageRankMutateProc;
import org.neo4j.graphalgo.wcc.WccMutateProc;

import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.neo4j.graphalgo.TestSupport.assertGraphEquals;
import static org.neo4j.graphalgo.TestSupport.fromGdl;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

class GraphMutateProcIntegrationTest extends BaseProcTest {

    private static final String TEST_GRAPH = "testGraph";

    private static final String DB_CYPHER =
        "CREATE" +
        "  (a:Node { nodeId: 0 })" +
        ", (b:Node { nodeId: 1 })" +
        ", (c:Node { nodeId: 2 })" +
        ", (d:Node { nodeId: 3 })" +
        ", (e:Node { nodeId: 4 })" +
        ", (f:Node { nodeId: 5 })" +
        ", (g:Node { nodeId: 6 })" +
        ", (h:Node { nodeId: 7 })" +
        ", (i:Node { nodeId: 8 })" +
        ", (j:Node { nodeId: 9 })" +
        ", (k:Node { nodeId: 10 })" +
        ", (l:Node { nodeId: 11 })" +
        ", (a)-[:TYPE {p: 10}]->(b)" +
        ", (b)-[:TYPE]->(c)" +
        ", (c)-[:TYPE]->(d)" +
        ", (d)-[:TYPE]->(e)" +
        ", (e)-[:TYPE]->(f)" +
        ", (f)-[:TYPE]->(g)" +
        ", (h)-[:TYPE]->(i)" +
        ", (i)-[:TYPE]->(k)" +
        ", (i)-[:TYPE]->(l)" +
        ", (j)-[:TYPE]->(k)";

    private static final Graph EXPECTED_GRAPH = fromGdl(
        "(a {nodeId: 0,  labelPropagation: 2,  louvain: 6,  pageRank: 0.150000, wcc: 0})" +
        "(b {nodeId: 1,  labelPropagation: 3,  louvain: 6,  pageRank: 0.277500, wcc: 0})" +
        "(c {nodeId: 2,  labelPropagation: 5,  louvain: 6,  pageRank: 0.385875, wcc: 0})" +
        "(d {nodeId: 3,  labelPropagation: 6,  louvain: 6,  pageRank: 0.477994, wcc: 0})" +
        "(e {nodeId: 4,  labelPropagation: 6,  louvain: 6,  pageRank: 0.556295, wcc: 0})" +
        "(f {nodeId: 5,  labelPropagation: 6,  louvain: 6,  pageRank: 0.622850, wcc: 0})" +
        "(g {nodeId: 6,  labelPropagation: 6,  louvain: 6,  pageRank: 0.679423, wcc: 0})" +
        "(h {nodeId: 7,  labelPropagation: 10, louvain: 10, pageRank: 0.150000, wcc: 7})" +
        "(i {nodeId: 8,  labelPropagation: 11, louvain: 10, pageRank: 0.277500, wcc: 7})" +
        "(j {nodeId: 9,  labelPropagation: 10, louvain: 10, pageRank: 0.150000, wcc: 7})" +
        "(k {nodeId: 10, labelPropagation: 10, louvain: 10, pageRank: 0.395438, wcc: 7})" +
        "(l {nodeId: 11, labelPropagation: 11, louvain: 11, pageRank: 0.267938, wcc: 7})" +
        "(a)-[{w: 1.0}]->(b)" +
        "(b)-[{w: 1.0}]->(c)" +
        "(c)-[{w: 1.0}]->(d)" +
        "(d)-[{w: 1.0}]->(e)" +
        "(e)-[{w: 1.0}]->(f)" +
        "(f)-[{w: 1.0}]->(g)" +
        "(h)-[{w: 1.0}]->(i)" +
        "(i)-[{w: 1.0}]->(k)" +
        "(i)-[{w: 1.0}]->(l)" +
        "(j)-[{w: 1.0}]->(k)" +
        // SIMILAR_TO
        "(i)-[{w: 0.5}]->(j)" +
        "(j)-[{w: 0.5}]->(i)"
    );

    @BeforeEach
    void setup() throws Exception {
        runQuery(DB_CYPHER);
        registerProcedures(
            GraphCreateProc.class,
            GraphDropProc.class,
            GraphWriteNodePropertiesProc.class,
            GraphWriteRelationshipProc.class,
            PageRankMutateProc.class,
            WccMutateProc.class,
            LabelPropagationMutateProc.class,
            LouvainMutateProc.class,
            NodeSimilarityMutateProc.class,
            GraphSageStreamProc.class
        );

        runQuery(GdsCypher
            .call()
            .withAnyLabel()
            .withNodeProperty("nodeId")
            .withRelationshipType("TYPE")
            .graphCreate(TEST_GRAPH)
            .yields());
    }

    @AfterEach
    void shutdown() {
        GraphStoreCatalog.removeAllLoadedGraphs();
    }

    @Test
    void runPipeline() {
        // run algorithms in mutate mode
        String pageRankQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("pageRank")
            .mutateMode()
            .addParameter("mutateProperty", "pageRank")
            .yields();
        String wccQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("wcc")
            .mutateMode()
            .addParameter("mutateProperty", "wcc")
            .yields();
        String labelPropagationQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("labelPropagation")
            .mutateMode()
            .addParameter("nodeWeightProperty", "pageRank")
            .addParameter("mutateProperty", "louvain")
            .yields();
        String louvainQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("louvain")
            .mutateMode()
            .addParameter("mutateProperty", "labelPropagation")
            .yields();
        String nodeSimilarityQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("nodeSimilarity")
            .mutateMode()
            .addParameter("mutateProperty", "similarity")
            .addParameter("mutateRelationshipType", "SIMILAR_TO")
            .yields();

        runQuery(pageRankQuery);
        runQuery(wccQuery);
        runQuery(labelPropagationQuery);
        runQuery(louvainQuery);
        runQuery(nodeSimilarityQuery);

        assertGraphEquals(EXPECTED_GRAPH, GraphStoreCatalog.get(getUsername(), db.databaseId(), TEST_GRAPH).graphStore().getUnion());

        int embeddingSize = 64;
        String graphSageQuery = GdsCypher
            .call()
            .explicitCreation(TEST_GRAPH)
            .algo("gds.alpha.graphSage")
            .streamMode()
            .addParameter("nodePropertyNames", List.of("pageRank", "louvain", "labelPropagation", "wcc"))
            .addParameter("embeddingSize", embeddingSize)
            .yields();

        runQueryWithRowConsumer(graphSageQuery, row -> {
            Collection<Double> embeddings = (Collection<Double>) row.get("embeddings");
            assertEquals(embeddings.size(), embeddingSize);

            double[] values = embeddings.stream()
                .mapToDouble(Double::doubleValue)
                .toArray();
            assertNotEquals(0D, L2Norm.l2(Tensor.vector(values)));
        });

        // write new properties and relationships to Neo
        String writeNodePropertiesQuery = formatWithLocale(
            "CALL gds.graph.writeNodeProperties(" +
            "   '%s', " +
            "   ['pageRank', 'louvain', 'labelPropagation', 'wcc']" +
            ")",
            TEST_GRAPH
        );
        runQueryWithRowConsumer(writeNodePropertiesQuery, row -> {
            assertEquals(row.getNumber("propertiesWritten").longValue(), 48L);
        });

        String writeRelationshipTypeQuery = formatWithLocale(
            "CALL gds.graph.writeRelationship(" +
            "   '%s', " +
            "   'SIMILAR_TO', " +
            "   'similarity'" +
            ")",
            TEST_GRAPH
        );
        runQuery(writeRelationshipTypeQuery);

        // re-create named graph from written node and relationship properties
        runQuery(formatWithLocale("CALL gds.graph.drop('%s')", TEST_GRAPH));
        runQuery(GdsCypher.call().withAnyLabel()
            .withNodeProperty("nodeId")
            .withNodeProperty("pageRank")
            .withNodeProperty("louvain")
            .withNodeProperty("wcc")
            .withNodeProperty("labelPropagation")
            .withRelationshipType("TYPE")
            .withRelationshipType("SIMILAR_TO")
            .withRelationshipProperty("similarity", 1.0)
            .graphCreate(TEST_GRAPH)
            .yields()
        );

        assertGraphEquals(EXPECTED_GRAPH, GraphStoreCatalog.get(getUsername(), db.databaseId(), TEST_GRAPH).graphStore().getUnion());
    }
}
