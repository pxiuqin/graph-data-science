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
package org.neo4j.gds.estimation.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
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
import org.neo4j.graphalgo.results.MemoryEstimateResult;
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

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.ALL_NODES_QUERY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.ALL_RELATIONSHIPS_QUERY;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.humanReadable;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

final class EstimationCliTest {

    private static final String PR_ESTIMATE = "gds.pageRank.stream.estimate";

    private static final String EXPECTED_JSON_TEMPLATE =
        "{\n" +
        "  \"bytes_min\" : %d,\n" +
        "  \"bytes_max\" : %d,\n" +
        "  \"min_memory\" : \"%s\",\n" +
        "  \"max_memory\" : \"%s\",\n" +
        "  \"procedure\" : \"%s\",\n" +
        "  \"node_count\" : 42,\n" +
        "  \"relationship_count\" : 1337,\n" +
        "  \"label_count\" : 0,\n" +
        "  \"relationship_type_count\" : 0,\n" +
        "  \"node_property_count\" : 0,\n" +
        "  \"relationship_property_count\" : 0\n" +
        "}";

    private static final List<String> PROCEDURES = List.of(
        "gds.beta.k1coloring.mutate.estimate",
        "gds.beta.k1coloring.stats.estimate",
        "gds.beta.k1coloring.stream.estimate",
        "gds.beta.k1coloring.write.estimate",

        "gds.beta.modularityOptimization.mutate.estimate",
        "gds.beta.modularityOptimization.stream.estimate",
        "gds.beta.modularityOptimization.write.estimate",

        "gds.betweenness.mutate.estimate",
        "gds.betweenness.stats.estimate",
        "gds.betweenness.stream.estimate",
        "gds.betweenness.write.estimate",

        "gds.graph.create.cypher.estimate",
        "gds.graph.create.estimate",

        "gds.labelPropagation.mutate.estimate",
        "gds.labelPropagation.stats.estimate",
        "gds.labelPropagation.stream.estimate",
        "gds.labelPropagation.write.estimate",

        "gds.localClusteringCoefficient.mutate.estimate",
        "gds.localClusteringCoefficient.stats.estimate",
        "gds.localClusteringCoefficient.stream.estimate",
        "gds.localClusteringCoefficient.write.estimate",

        "gds.louvain.mutate.estimate",
        "gds.louvain.stats.estimate",
        "gds.louvain.stream.estimate",
        "gds.louvain.write.estimate",

        "gds.nodeSimilarity.mutate.estimate",
        "gds.nodeSimilarity.stats.estimate",
        "gds.nodeSimilarity.stream.estimate",
        "gds.nodeSimilarity.write.estimate",

        "gds.pageRank.mutate.estimate",
        "gds.pageRank.stats.estimate",
        "gds.pageRank.stream.estimate",
        "gds.pageRank.write.estimate",

        "gds.triangleCount.mutate.estimate",
        "gds.triangleCount.stats.estimate",
        "gds.triangleCount.stream.estimate",
        "gds.triangleCount.write.estimate",

        "gds.wcc.mutate.estimate",
        "gds.wcc.stats.estimate",
        "gds.wcc.stream.estimate",
        "gds.wcc.write.estimate"
    );


    @ParameterizedTest
    @CsvSource({
        "--nodes, --relationships",
        "-n, -r",
    })
    void runsEstimation(String nodeArg, String relArg) {
        var actual = run(PR_ESTIMATE, nodeArg, 42, relArg, 1337);
        var expected = pageRankEstimate();

        assertEquals("gds.pagerank.stream.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @ParameterizedTest
    @ValueSource(strings = {"--labels", "-l"})
    void runsEstimationWithLabels(String labelsArg) {
        var actual = run(PR_ESTIMATE, "--nodes", 42, "--relationships", 1337, labelsArg, 21);
        var expected = pageRankEstimate("nodeProjection", listOfIdentifiers(21));

        assertEquals("gds.pagerank.stream.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @ParameterizedTest
    @ValueSource(strings = {"--node-properties", "-np"})
    void runsEstimationWithNodeProperties(String nodePropsArg) {
        var actual = run(PR_ESTIMATE, "--nodes", 42, "--relationships", 1337, nodePropsArg, 21);
        var expected = pageRankEstimate("nodeProperties", listOfIdentifiers(21));

        assertEquals("gds.pagerank.stream.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @ParameterizedTest
    @ValueSource(strings = {"--relationship-properties", "-rp"})
    void runsEstimationWithRelationshipProperties(String relPropsArg) {
        var actual = run(PR_ESTIMATE, "--nodes", 42, "--relationships", 1337, relPropsArg, 21);
        var expected = pageRankEstimate("relationshipProperties", listOfIdentifiers(21));

        assertEquals("gds.pagerank.stream.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @Test
    void runsEstimationWithConcurrency() {
        var actual = run(PR_ESTIMATE, "--nodes", 42, "--relationships", 1337, "-c", "readConcurrency=21");
        var expected = pageRankEstimate("readConcurrency", 21);

        assertEquals("gds.pagerank.stream.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @Test
    void estimatesGraphCreate() {
        var actual = run("gds.graph.create", "--nodes", 42, "--relationships", 1337);
        var expected = graphCreateEstimate(false);

        assertEquals("gds.graph.create.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @Test
    void estimatesGraphCreateCypher() {
        var actual = run("gds.graph.create.cypher", "--nodes", 42, "--relationships", 1337);
        var expected = graphCreateEstimate(true);

        assertEquals("gds.graph.create.cypher.estimate," + expected.bytesMin + "," + expected.bytesMax, actual);
    }

    @Test
    void printsTree() {
        var actual = run(PR_ESTIMATE, "-n", 42, "-r", 1337, "--tree");
        var expected = pageRankEstimate();

        assertEquals("gds.pagerank.stream.estimate," + expected.treeView.strip(), actual);
    }

    @Test
    void printsJson() {
        var actual = run(PR_ESTIMATE, "-n", 42, "-r", 1337, "--json");
        var expected = pageRankEstimate();
        var expectedJson = expectedJson(expected, "gds.pagerank.stream.estimate");

        assertEquals(expectedJson, actual);
    }

    @Test
    void nodeCountIsMandatory() {
        var actual = assertThrows(ExecutionFailed.class, () -> run(PR_ESTIMATE, "-r", 1337));

        assertEquals(2, actual.exitCode);
        assertEquals(
            "Missing required option: '--nodes=<nodeCount>'",
            actual.stderr.lines().iterator().next()
        );
    }

    @Test
    void relationshipCountIsMandatory() {
        var actual = assertThrows(ExecutionFailed.class, () -> run(PR_ESTIMATE, "-n", 42));

        assertEquals(2, actual.exitCode);
        assertEquals(
            "Missing required option: '--relationships=<relationshipCount>'",
            actual.stderr.lines().iterator().next()
        );
    }

    @Test
    void cannotPrintTreeAndJson() {
        var actual = assertThrows(
            ExecutionFailed.class,
            () -> run(PR_ESTIMATE, "-n", 42, "-r", 1337, "--json", "--tree")
        );

        assertEquals(2, actual.exitCode);
        assertEquals(
            "Error: --tree, --json are mutually exclusive (specify only one)",
            actual.stderr.lines().iterator().next()
        );
    }

    @Test
    void listAllAvailableProcedures() {
        var actual = run("list-available");
        var expected = PROCEDURES.stream().collect(joining(System.lineSeparator()));

        assertEquals(expected, actual);
    }

    @Test
    void estimateAllAvailableProcedures() {
        Stream<MemoryEstimateResult> expectedEstimations = allEstimations();

        var expectedProcedureNames = PROCEDURES.iterator();
        var expected = expectedEstimations
            .map(e -> expectedProcedureNames.next() + "," + e.bytesMin + "," + e.bytesMax)
            .collect(joining(System.lineSeparator()));

        var actual = run("-n", 42, "-r", 1337);
        assertEquals(expected, actual);
    }

    @Test
    void estimateAllAvailableProceduresInTreeMode() {
        Stream<MemoryEstimateResult> expectedEstimations = allEstimations();

        var expectedProcedureNames = PROCEDURES.iterator();
        var expected = expectedEstimations
            .map(e -> expectedProcedureNames.next() + "," + e.treeView)
            .collect(joining(System.lineSeparator()));

        var actual = run("-n", 42, "-r", 1337, "--tree");
        assertEquals(expected.strip(), actual);
    }

    @Test
    void estimateAllAvailableProceduresInTreeInJsonMode() {
        Stream<MemoryEstimateResult> expectedEstimations = allEstimations();

        var expectedProcedureNames = PROCEDURES.iterator();
        var expected = expectedEstimations
            .map(e -> expectedJson(e, expectedProcedureNames.next()))
            .collect(joining(", ", "[ ", " ]"));

        var actual = run("-n", 42, "-r", 1337, "--json");
        assertEquals(expected, actual);
    }

    private String expectedJson(MemoryEstimateResult expected, String procName) {
        return formatWithLocale(
            EXPECTED_JSON_TEMPLATE,
            expected.bytesMin,
            expected.bytesMax,
            humanReadable(expected.bytesMin),
            humanReadable(expected.bytesMax),
            procName
        );
    }

    private static Stream<MemoryEstimateResult> allEstimations() {
        return Stream.of(
            runEstimation(new K1ColoringMutateProc()::mutateEstimate, "mutateProperty", "foo"),
            runEstimation(new K1ColoringStatsProc()::estimate),
            runEstimation(new K1ColoringStreamProc()::estimate),
            runEstimation(new K1ColoringWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(new ModularityOptimizationMutateProc()::mutateEstimate, "mutateProperty", "foo"),
            runEstimation(new ModularityOptimizationStreamProc()::estimate),
            runEstimation(new ModularityOptimizationWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(new BetweennessCentralityMutateProc()::estimate, "mutateProperty", "foo"),
            runEstimation(new BetweennessCentralityStatsProc()::estimate),
            runEstimation(new BetweennessCentralityStreamProc()::estimate),
            runEstimation(new BetweennessCentralityWriteProc()::estimate, "writeProperty", "foo"),

            graphCreateEstimate(false),
            graphCreateEstimate(true),

            runEstimation(new LabelPropagationMutateProc()::mutateEstimate, "mutateProperty", "foo"),
            runEstimation(new LabelPropagationStatsProc()::estimateStats),
            runEstimation(new LabelPropagationStreamProc()::estimate),
            runEstimation(new LabelPropagationWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(new LocalClusteringCoefficientMutateProc()::estimate, "mutateProperty", "foo"),
            runEstimation(new LocalClusteringCoefficientStatsProc()::estimateStats),
            runEstimation(new LocalClusteringCoefficientStreamProc()::estimateStats),
            runEstimation(new LocalClusteringCoefficientWriteProc()::estimateStats, "writeProperty", "foo"),

            runEstimation(new LouvainMutateProc()::estimate, "mutateProperty", "foo"),
            runEstimation(new LouvainStatsProc()::estimateStats),
            runEstimation(new LouvainStreamProc()::estimate),
            runEstimation(new LouvainWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(
                new NodeSimilarityMutateProc()::estimateMutate,
                "mutateProperty",
                "foo",
                "mutateRelationshipType",
                "bar"
            ),
            runEstimation(new NodeSimilarityStatsProc()::estimateStats),
            runEstimation(new NodeSimilarityStreamProc()::estimate),
            runEstimation(
                new NodeSimilarityWriteProc()::estimateWrite,
                "writeProperty",
                "foo",
                "writeRelationshipType",
                "bar"
            ),

            runEstimation(new PageRankMutateProc()::estimate, "mutateProperty", "foo"),
            runEstimation(new PageRankStatsProc()::estimateStats),
            runEstimation(new PageRankStreamProc()::estimate),
            runEstimation(new PageRankWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(new TriangleCountMutateProc()::estimate, "mutateProperty", "foo"),
            runEstimation(new TriangleCountStatsProc()::estimateStats),
            runEstimation(new TriangleCountStreamProc()::estimateStats),
            runEstimation(new TriangleCountWriteProc()::estimate, "writeProperty", "foo"),

            runEstimation(new WccMutateProc()::mutateEstimate, "mutateProperty", "foo"),
            runEstimation(new WccStatsProc()::statsEstimate),
            runEstimation(new WccStreamProc()::streamEstimate),
            runEstimation(new WccWriteProc()::writeEstimate, "writeProperty", "foo")
        );
    }

    private static final class ExecutionFailed extends RuntimeException {
        final int exitCode;
        final String stderr;

        ExecutionFailed(int exitCode, String stderr) {
            super(formatWithLocale(
                "Calling CLI failed with exit code %d and stderr: %s",
                exitCode,
                stderr
            ));
            this.exitCode = exitCode;
            this.stderr = stderr;
        }
    }

    private static String run(Object... args) {
        var arguments = new String[args.length];
        Arrays.setAll(arguments, i -> String.valueOf(args[i]));

        var stdout = new ByteArrayOutputStream(8192);
        var stderr = new ByteArrayOutputStream(8192);
        var originalOut = System.out;
        var originalErr = System.err;
        var exitCode = -1;

        try {

            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            System.setErr(new PrintStream(stderr, true, StandardCharsets.UTF_8));

            exitCode = EstimationCli.runWithArgs(arguments);

        } finally {
            System.setErr(originalErr);
            System.setOut(originalOut);
        }


        if (exitCode != 0) {
            throw new ExecutionFailed(exitCode, stderr.toString(StandardCharsets.UTF_8));
        }

        return stdout.toString(StandardCharsets.UTF_8).strip();
    }

    private static MemoryEstimateResult pageRankEstimate(Object... config) {
        return runEstimation(new PageRankStreamProc()::estimate, config);
    }

    private static MemoryEstimateResult runEstimation(
        BiFunction<Object, Map<String, Object>, Stream<MemoryEstimateResult>> proc,
        Object... config
    ) {
        Map<String, Object> configMap = new HashMap<>(Map.of(
            "nodeCount", 42L,
            "relationshipCount", 1337L,
            "nodeProjection", "*",
            "relationshipProjection", "*"
        ));
        for (int i = 0; i < config.length; i += 2) {
            configMap.put(String.valueOf(config[i]), config[i + 1]);
        }
        return proc.apply(configMap, Map.of()).iterator().next();
    }

    private static MemoryEstimateResult graphCreateEstimate(boolean cypher) {
        Map<String, Object> config = Map.of(
            "nodeCount", 42L,
            "relationshipCount", 1337L
        );

        var gc = new GraphCreateProc();
        var result = cypher
            ? gc.createCypherEstimate(ALL_NODES_QUERY, ALL_RELATIONSHIPS_QUERY, config)
            : gc.createEstimate("*", "*", config);

        return result.iterator().next();
    }

    private static List<String> listOfIdentifiers(int numberOfIdentifiers) {
        return IntStream
            .range(0, numberOfIdentifiers)
            .mapToObj(i -> String.valueOf((char) ('A' + i)))
            .collect(toList());
    }

}
