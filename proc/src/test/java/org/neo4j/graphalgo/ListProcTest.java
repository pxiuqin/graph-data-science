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
package org.neo4j.graphalgo;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.beta.generator.GraphGenerateProc;
import org.neo4j.graphalgo.beta.k1coloring.K1ColoringMutateProc;
import org.neo4j.graphalgo.beta.k1coloring.K1ColoringStatsProc;
import org.neo4j.graphalgo.beta.k1coloring.K1ColoringStreamProc;
import org.neo4j.graphalgo.beta.k1coloring.K1ColoringWriteProc;
import org.neo4j.graphalgo.beta.modularity.ModularityOptimizationMutateProc;
import org.neo4j.graphalgo.beta.modularity.ModularityOptimizationStreamProc;
import org.neo4j.graphalgo.beta.modularity.ModularityOptimizationWriteProc;
import org.neo4j.graphalgo.betweenness.BetweennessCentralityMutateProc;
import org.neo4j.graphalgo.betweenness.BetweennessCentralityStatsProc;
import org.neo4j.graphalgo.betweenness.BetweennessCentralityStreamProc;
import org.neo4j.graphalgo.betweenness.BetweennessCentralityWriteProc;
import org.neo4j.graphalgo.catalog.GraphCreateProc;
import org.neo4j.graphalgo.catalog.GraphDeleteRelationshipProc;
import org.neo4j.graphalgo.catalog.GraphDropProc;
import org.neo4j.graphalgo.catalog.GraphExistsProc;
import org.neo4j.graphalgo.catalog.GraphListProc;
import org.neo4j.graphalgo.catalog.GraphStreamNodePropertiesProc;
import org.neo4j.graphalgo.catalog.GraphStreamRelationshipPropertiesProc;
import org.neo4j.graphalgo.catalog.GraphWriteNodePropertiesProc;
import org.neo4j.graphalgo.compat.MapUtil;
import org.neo4j.graphalgo.functions.AsNodeFunc;
import org.neo4j.graphalgo.functions.VersionFunc;
import org.neo4j.graphalgo.labelpropagation.LabelPropagationMutateProc;
import org.neo4j.graphalgo.labelpropagation.LabelPropagationStatsProc;
import org.neo4j.graphalgo.labelpropagation.LabelPropagationStreamProc;
import org.neo4j.graphalgo.labelpropagation.LabelPropagationWriteProc;
import org.neo4j.graphalgo.louvain.LouvainMutateProc;
import org.neo4j.graphalgo.louvain.LouvainStatsProc;
import org.neo4j.graphalgo.louvain.LouvainStreamProc;
import org.neo4j.graphalgo.louvain.LouvainWriteProc;
import org.neo4j.graphalgo.nodesim.NodeSimilarityMutateProc;
import org.neo4j.graphalgo.nodesim.NodeSimilarityStatsProc;
import org.neo4j.graphalgo.nodesim.NodeSimilarityStreamProc;
import org.neo4j.graphalgo.nodesim.NodeSimilarityWriteProc;
import org.neo4j.graphalgo.pagerank.PageRankMutateProc;
import org.neo4j.graphalgo.pagerank.PageRankStatsProc;
import org.neo4j.graphalgo.pagerank.PageRankStreamProc;
import org.neo4j.graphalgo.pagerank.PageRankWriteProc;
import org.neo4j.graphalgo.triangle.LocalClusteringCoefficientMutateProc;
import org.neo4j.graphalgo.triangle.LocalClusteringCoefficientStatsProc;
import org.neo4j.graphalgo.triangle.LocalClusteringCoefficientStreamProc;
import org.neo4j.graphalgo.triangle.LocalClusteringCoefficientWriteProc;
import org.neo4j.graphalgo.triangle.TriangleCountMutateProc;
import org.neo4j.graphalgo.triangle.TriangleCountStatsProc;
import org.neo4j.graphalgo.triangle.TriangleCountStreamProc;
import org.neo4j.graphalgo.triangle.TriangleCountWriteProc;
import org.neo4j.graphalgo.wcc.WccMutateProc;
import org.neo4j.graphalgo.wcc.WccStatsProc;
import org.neo4j.graphalgo.wcc.WccStreamProc;
import org.neo4j.graphalgo.wcc.WccWriteProc;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ListProcTest extends BaseProcTest {

    private static final List<String> PROCEDURES = asList(
        "gds.beta.graph.generate",

        "gds.beta.k1coloring.mutate",
        "gds.beta.k1coloring.mutate.estimate",
        "gds.beta.k1coloring.stats",
        "gds.beta.k1coloring.stats.estimate",
        "gds.beta.k1coloring.stream",
        "gds.beta.k1coloring.stream.estimate",
        "gds.beta.k1coloring.write",
        "gds.beta.k1coloring.write.estimate",

        "gds.beta.modularityOptimization.mutate",
        "gds.beta.modularityOptimization.mutate.estimate",
        "gds.beta.modularityOptimization.stream",
        "gds.beta.modularityOptimization.stream.estimate",
        "gds.beta.modularityOptimization.write",
        "gds.beta.modularityOptimization.write.estimate",

        "gds.betweenness.mutate",
        "gds.betweenness.mutate.estimate",
        "gds.betweenness.stats",
        "gds.betweenness.stats.estimate",
        "gds.betweenness.stream",
        "gds.betweenness.stream.estimate",
        "gds.betweenness.write",
        "gds.betweenness.write.estimate",

        "gds.graph.create",
        "gds.graph.create.cypher",
        "gds.graph.create.cypher.estimate",
        "gds.graph.create.estimate",
        "gds.graph.deleteRelationships",
        "gds.graph.drop",
        "gds.graph.exists",
        "gds.graph.list",
        "gds.graph.streamNodeProperties",
        "gds.graph.streamNodeProperty",
        "gds.graph.streamRelationshipProperties",
        "gds.graph.streamRelationshipProperty",
        "gds.graph.writeNodeProperties",

        "gds.labelPropagation.mutate",
        "gds.labelPropagation.mutate.estimate",
        "gds.labelPropagation.stats",
        "gds.labelPropagation.stats.estimate",
        "gds.labelPropagation.stream",
        "gds.labelPropagation.stream.estimate",
        "gds.labelPropagation.write",
        "gds.labelPropagation.write.estimate",

        "gds.louvain.mutate",
        "gds.louvain.mutate.estimate",
        "gds.louvain.stats",
        "gds.louvain.stats.estimate",
        "gds.louvain.stream",
        "gds.louvain.stream.estimate",
        "gds.louvain.write",
        "gds.louvain.write.estimate",

        "gds.nodeSimilarity.mutate",
        "gds.nodeSimilarity.mutate.estimate",
        "gds.nodeSimilarity.stats",
        "gds.nodeSimilarity.stats.estimate",
        "gds.nodeSimilarity.stream",
        "gds.nodeSimilarity.stream.estimate",
        "gds.nodeSimilarity.write",
        "gds.nodeSimilarity.write.estimate",

        "gds.pageRank.mutate",
        "gds.pageRank.mutate.estimate",
        "gds.pageRank.stats",
        "gds.pageRank.stats.estimate",
        "gds.pageRank.stream",
        "gds.pageRank.stream.estimate",
        "gds.pageRank.write",
        "gds.pageRank.write.estimate",

        "gds.wcc.mutate",
        "gds.wcc.mutate.estimate",
        "gds.wcc.stats",
        "gds.wcc.stats.estimate",
        "gds.wcc.stream",
        "gds.wcc.stream.estimate",
        "gds.wcc.write",
        "gds.wcc.write.estimate",

        "gds.triangleCount.mutate",
        "gds.triangleCount.mutate.estimate",
        "gds.triangleCount.stats",
        "gds.triangleCount.stats.estimate",
        "gds.triangleCount.stream",
        "gds.triangleCount.stream.estimate",
        "gds.triangleCount.write",
        "gds.triangleCount.write.estimate",

        "gds.localClusteringCoefficient.mutate",
        "gds.localClusteringCoefficient.mutate.estimate",
        "gds.localClusteringCoefficient.stats",
        "gds.localClusteringCoefficient.stats.estimate",
        "gds.localClusteringCoefficient.stream",
        "gds.localClusteringCoefficient.stream.estimate",
        "gds.localClusteringCoefficient.write",
        "gds.localClusteringCoefficient.write.estimate"
    );

    private static final List<String> FUNCTIONS = asList(
        "gds.util.asNode",
        "gds.util.asNodes",
        "gds.version"
    );

    private static final List<String> PAGE_RANK = asList(
        "gds.pageRank.mutate",
        "gds.pageRank.mutate.estimate",
        "gds.pageRank.stats",
        "gds.pageRank.stats.estimate",
        "gds.pageRank.stream",
        "gds.pageRank.stream.estimate",
        "gds.pageRank.write",
        "gds.pageRank.write.estimate"
    );

    private static final List<String> ALL = Stream
        .concat(PROCEDURES.stream(), FUNCTIONS.stream())
        .sorted()
        .collect(Collectors.toList());

    @BeforeEach
    void setUp() throws Exception {
        registerProcedures(
            BetweennessCentralityStreamProc.class,
            BetweennessCentralityWriteProc.class,
            BetweennessCentralityMutateProc.class,
            BetweennessCentralityStatsProc.class,
            GraphCreateProc.class,
            GraphDropProc.class,
            GraphExistsProc.class,
            GraphListProc.class,
            GraphGenerateProc.class,
            GraphDeleteRelationshipProc.class,
            GraphStreamNodePropertiesProc.class,
            GraphStreamRelationshipPropertiesProc.class,
            GraphWriteNodePropertiesProc.class,
            K1ColoringMutateProc.class,
            K1ColoringStatsProc.class,
            K1ColoringWriteProc.class,
            K1ColoringStreamProc.class,
            LabelPropagationWriteProc.class,
            LabelPropagationStreamProc.class,
            LabelPropagationStatsProc.class,
            LabelPropagationMutateProc.class,
            ListProc.class,
            LouvainWriteProc.class,
            LouvainStreamProc.class,
            LouvainStatsProc.class,
            LouvainMutateProc.class,
            ModularityOptimizationMutateProc.class,
            ModularityOptimizationWriteProc.class,
            ModularityOptimizationStreamProc.class,
            NodeSimilarityWriteProc.class,
            NodeSimilarityStreamProc.class,
            NodeSimilarityMutateProc.class,
            NodeSimilarityStatsProc.class,
            PageRankWriteProc.class,
            PageRankStreamProc.class,
            PageRankMutateProc.class,
            PageRankStatsProc.class,
            TriangleCountStatsProc.class,
            TriangleCountWriteProc.class,
            TriangleCountStreamProc.class,
            TriangleCountMutateProc.class,
            WccWriteProc.class,
            WccStreamProc.class,
            WccMutateProc.class,
            WccStatsProc.class,
            LocalClusteringCoefficientStreamProc.class,
            LocalClusteringCoefficientStatsProc.class,
            LocalClusteringCoefficientWriteProc.class,
            LocalClusteringCoefficientMutateProc.class
        );
        registerFunctions(
            AsNodeFunc.class,
            VersionFunc.class
        );
    }

    @Test
    void shouldListAllThingsExceptTheListProcedure() {
        assertEquals(ALL, listProcs(null));
    }

    @Test
    void listFilteredResult() {
        assertEquals(PAGE_RANK, listProcs("pageRank"));
        assertEquals(asList("gds.pageRank.stream", "gds.pageRank.stream.estimate"), listProcs("pageRank.stream"));
        assertEquals(emptyList(), listProcs("foo"));
    }

    @Test
    void listFunctions() {
        List<String> actual = listProcs("asNode");
        actual.addAll(listProcs("getNode"));
        actual.addAll(listProcs("version"));
        assertEquals(FUNCTIONS, actual);
    }

    @Test
    void listEmpty() {
        String query = "CALL gds.list()";
        assertEquals(
            ALL,
            runQuery(query, result -> result
                .<String>columnAs("name")
                .stream()
                .sorted()
                .collect(Collectors.toList())));
    }

    private List<String> listProcs(Object name) {
        String query = "CALL gds.list($name)";
        return runQuery(
            query,
            MapUtil.map("name", name),
            result -> result.<String>columnAs("name")
                .stream()
                .sorted()
                .collect(Collectors.toList())
        );
    }
}
