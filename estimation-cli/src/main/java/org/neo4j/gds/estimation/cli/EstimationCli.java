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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.jetbrains.annotations.NotNull;
import org.neo4j.graphalgo.ElementProjection;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.config.GraphCreateFromCypherConfig;
import org.neo4j.graphalgo.core.GdsEdition;
import org.neo4j.graphalgo.results.MemoryEstimateResult;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import picocli.CommandLine;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.function.Predicate.isEqual;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.joining;
import static org.neo4j.graphalgo.config.GraphCreateConfig.NODE_COUNT_KEY;
import static org.neo4j.graphalgo.config.GraphCreateConfig.RELATIONSHIP_COUNT_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.NODE_QUERY_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromCypherConfig.RELATIONSHIP_QUERY_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.NODE_PROPERTIES_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROJECTION_KEY;
import static org.neo4j.graphalgo.config.GraphCreateFromStoreConfig.RELATIONSHIP_PROPERTIES_KEY;
import static org.neo4j.graphalgo.config.MutatePropertyConfig.MUTATE_PROPERTY_KEY;
import static org.neo4j.graphalgo.config.WritePropertyConfig.WRITE_PROPERTY_KEY;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.humanReadable;
import static org.neo4j.graphalgo.utils.CheckedFunction.function;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

@SuppressWarnings({"FieldMayBeFinal", "FieldCanBeLocal", "DefaultAnnotationParam"})
@CommandLine.Command(
    description = "Estimates the memory consumption of a GDS procedure.",
    name = "estimation-cli",
    mixinStandardHelpOptions = true,
    version = "estimation-cli 0.4.2"
)
public class EstimationCli implements Runnable {

    @CommandLine.Spec
    CommandLine.Model.CommandSpec spec;

    @CommandLine.Command(name = "list-available")
    void listAvailable() {
        var availableProcedures = findAvailableMethods()
            .map(ProcedureMethod::name)
            .collect(joining(System.lineSeparator()));

        System.out.println(availableProcedures);
    }

    @CommandLine.Command(name = "estimate")
    void estimateOne(
        @CommandLine.Parameters(
            paramLabel = "procedure",
            description = "The procedure to estimate, e.g. gds.pagerank.stream.",
            converter = ProcedureNameNormalizer.class,
            defaultValue = ""
        )
            String procedureName,

        @CommandLine.Mixin
            CountOptions counts,

        @CommandLine.ArgGroup(exclusive = true)
            PrintOptions printOptions

    ) throws Exception {
        GdsEdition.instance().setToEnterpriseEdition();
        var printOpts = printOptions == null ? new PrintOptions() : printOptions;

        var procedureMethods = procedureName.isBlank()
            ? findAvailableMethods()
            : Stream.of(ImmutableProcedureMethod.of(procedureName, findProcedure(procedureName)));

        var estimations = procedureMethods
            .map(function(proc -> estimateProcedure(proc.name(), proc.method(), counts)))
            .collect(Collectors.toList());
        renderResults(counts, printOpts, estimations);
    }

    static final class CountOptions {
        @CommandLine.Option(
            names = {"-n", "--nodes"},
            description = "Number of nodes in the fictitious graph.",
            required = true,
            converter = LongParser.class
        )
        private long nodeCount;

        @CommandLine.Option(
            names = {"-r", "--relationships"},
            description = "Number of relationships in the fictitious graph.",
            required = true,
            converter = LongParser.class
        )
        private long relationshipCount;

        @CommandLine.Option(
            names = {"-l", "--labels"},
            description = "Number of node labels in the fictitious graph.",
            converter = IntParser.class
        )
        private int labelCount = 0;

        // We don't make use of this because the number of types does not influence the estimation.
        // We specify it here so that the options look symmetric, the result just doesn't change.
        @CommandLine.Option(
            names = {"-t", "--types"},
            description = "Number of relationship types in the fictitious graph.",
            converter = IntParser.class
        )
        private int relationshipTypes = 0;

        @CommandLine.Option(
            names = {"-np", "--node-properties"},
            description = "Number of node properties in the fictitious graph.",
            converter = IntParser.class
        )
        private int nodePropertyCount = 0;

        @CommandLine.Option(
            names = {"-rp", "--relationship-properties"},
            description = "Number of relationship properties in the fictitious graph.",
            converter = IntParser.class
        )
        private int relationshipPropertyCount = 0;

        @CommandLine.Option(
            names = {"-c", "--config"},
            description = "Numeric configuration options of the given procedure.",
            split = ","
        )
        private Map<String, Number> config;

        @NotNull
        private Map<String, Object> procedureConfig(String procedureName) {
            HashMap<String, Object> actualConfig;
            if (config != null) {
                actualConfig = new HashMap<>(config);
            } else {
                actualConfig = new HashMap<>();
            }

            actualConfig.put(NODE_COUNT_KEY, nodeCount);
            actualConfig.put(RELATIONSHIP_COUNT_KEY, relationshipCount);
            actualConfig.put(NODE_PROJECTION_KEY, labelCount > 0
                ? createEntries(labelCount, "Label")
                : ElementProjection.PROJECT_ALL
            );
            actualConfig.put(RELATIONSHIP_PROJECTION_KEY, ElementProjection.PROJECT_ALL);

            if (nodePropertyCount > 0) {
                actualConfig.put(NODE_PROPERTIES_KEY, createEntries(nodePropertyCount, "prop"));
            }
            if (relationshipPropertyCount > 0) {
                actualConfig.put(RELATIONSHIP_PROPERTIES_KEY, createEntries(relationshipPropertyCount, "prop"));
            }
            if (procedureName.endsWith(".write.estimate")) {
                actualConfig.put(WRITE_PROPERTY_KEY, "ESTIMATE_FAKE_WRITE_PROPERTY");
            }
            if (procedureName.endsWith(".mutate.estimate")) {
                actualConfig.put(MUTATE_PROPERTY_KEY, "ESTIMATE_FAKE_MUTATE_PROPERTY");
            }
            if (procedureName.equals("gds.nodeSimilarity.write.estimate")) {
                actualConfig.put("writeRelationshipType", "ESTIMATE_FAKE_WRITE_RELATIONSHIP_PROPERTY");
            }
            if (procedureName.equals("gds.nodeSimilarity.mutate.estimate")) {
                actualConfig.put("mutateRelationshipType", "ESTIMATE_FAKE_MUTATE_RELATIONSHIP_PROPERTY");
            }
            return actualConfig;
        }

        private List<String> createEntries(int count, String prefix) {
            return IntStream.range(0, count)
                .mapToObj(i -> formatWithLocale("%s%d", prefix, i))
                .collect(Collectors.toList());
        }
    }

    static final class PrintOptions {
        @CommandLine.Option(
            names = {"--tree"},
            description = "Print estimated memory as human readable tree view.",
            required = true
        )
        boolean printTree;

        @CommandLine.Option(
            names = {"--json"},
            description = "Print estimated memory in json format.",
            required = true
        )
        boolean printJson;
    }

    public static void main(String... args) {
        int exitCode = runWithArgs(args);
        System.exit(exitCode);
    }

    @Override
    public void run() {
        throw new CommandLine.ParameterException(spec.commandLine(), "Specify a subcommand");
    }

    static int runWithArgs(String... args) {
        var commandLine = new CommandLine(new EstimationCli())
            .registerConverter(Number.class, new NumberParser());

        boolean parsed;
        try {
            parsed = commandLine.parseArgs(args).hasSubcommand();
        } catch (RuntimeException e) {
            parsed = false;
        }

        if (!parsed) {
            var newArgs = new String[args.length + 1];
            System.arraycopy(args, 0, newArgs, 1, args.length);
            newArgs[0] = "estimate";
            args = newArgs;
        }

        return commandLine.execute(args);
    }

    private static final List<String> PACKAGES_TO_SCAN = List.of(
        "org.neo4j.graphalgo",
        "org.neo4j.gds.embeddings"
    );

    private Method findProcedure(String procedure) {
        return PACKAGES_TO_SCAN.stream()
            .map(pkg -> new Reflections(pkg, new MethodAnnotationsScanner()))
            .map(reflections -> findProcedure(reflections, procedure))
            .flatMap(Optional::stream)
            .findFirst()
            .orElseThrow(() -> new CommandLine.ParameterException(
                spec.commandLine(),
                formatWithLocale("Procedure not found: %s", procedure),
                spec.findOption("procedure"),
                procedure
            ));
    }

    private Optional<Method> findProcedure(Reflections reflections, String procedureName) {
        return reflections
            .getMethodsAnnotatedWith(Procedure.class)
            .stream()
            .filter(method -> {
                var annotation = method.getDeclaredAnnotation(Procedure.class);
                return annotation.value().equalsIgnoreCase(procedureName) ||
                       annotation.name().equalsIgnoreCase(procedureName);
            })
            .findFirst();
    }

    private Stream<ProcedureMethod> findAvailableMethods() {
        return PACKAGES_TO_SCAN.stream()
            .map(pkg -> new Reflections(pkg, new MethodAnnotationsScanner()))
            .flatMap(reflections -> reflections
                .getMethodsAnnotatedWith(Procedure.class)
                .stream())
            .flatMap(method -> {
                var annotation = method.getAnnotation(Procedure.class);
                var valueName = annotation.value();
                var definedName = annotation.name();
                var procName = definedName.trim().isEmpty() ? valueName : definedName;
                return Stream.of(procName)
                    .filter(s -> s.endsWith(".estimate"))
                    .filter(not(isEqual("gds.testProc.test.estimate")))
                    .map(name -> ImmutableProcedureMethod.of(name, method));
            })
            .sorted(Comparator.comparing(ProcedureMethod::name, String.CASE_INSENSITIVE_ORDER));
    }

    private EstimatedProcedure estimateProcedure(
        String procedureName,
        Method procedure,
        CountOptions counts
    ) throws Exception {
        var config = counts.procedureConfig(procedureName);
        var estimateResult = runProcedure(procedure, config);
        return ImmutableEstimatedProcedure.of(procedureName, estimateResult);
    }

    private static MemoryEstimateResult runProcedure(Method procedure, Map<String, Object> config) throws Exception {
        var parameters = procedure.getParameters();
        var args = new Object[parameters.length];
        var foundConfigParam = false;
        for (int i = 0; i < parameters.length; i++) {
            Parameter parameter = parameters[i];
            var name = parameter.getAnnotation(Name.class);
            if (name != null) {
                switch (name.value()) {
                    case NODE_QUERY_KEY:
                        args[i] = GraphCreateFromCypherConfig.ALL_NODES_QUERY;
                        config.remove(NODE_PROJECTION_KEY);
                        break;
                    case RELATIONSHIP_QUERY_KEY:
                        args[i] = GraphCreateFromCypherConfig.ALL_RELATIONSHIPS_QUERY;
                        config.remove(RELATIONSHIP_PROJECTION_KEY);
                        break;
                    case NODE_PROJECTION_KEY: // explicit fall-through
                    case RELATIONSHIP_PROJECTION_KEY:
                        args[i] = ElementProjection.PROJECT_ALL;
                        break;
                    case "graphName":
                        if (!foundConfigParam) {
                            foundConfigParam = true;
                            args[i] = config;
                        } else {
                            throw new RuntimeException(
                                "found parameter annotated with `graphName`, expected to accept a config object, but there was already another parameter that accepted the config."
                            );
                        }
                        break;
                    case "configuration":
                        if (!foundConfigParam) {
                            foundConfigParam = true;
                            args[i] = config;
                        } else {
                            args[i] = Map.of();
                        }
                        break;
                    default:
                        throw new RuntimeException(
                            "Unexpected parameter name: `" + name.value() + "`. This is probably a bug in GDS."
                        );
                }
            }
        }

        var procInstance = procedure.getDeclaringClass().getConstructor().newInstance();
        var procResultStream = (Stream<?>) procedure.invoke(procInstance, args);
        return (MemoryEstimateResult) procResultStream.findFirst().orElseThrow();
    }

    private static void renderResults(
        CountOptions countOptions,
        PrintOptions printOptions,
        Collection<EstimatedProcedure> estimatedProcedures
    ) throws IOException {
        if (printOptions.printTree) {
            for (EstimatedProcedure estimatedProcedure : estimatedProcedures) {
                System.out.printf(
                    Locale.ENGLISH,
                    "%s,%s%n",
                    estimatedProcedure.name(),
                    estimatedProcedure.estimation().treeView
                );
            }
        } else if (!printOptions.printJson) {
            for (EstimatedProcedure estimatedProcedure : estimatedProcedures) {
                var estimation = estimatedProcedure.estimation();
                System.out.printf(
                    Locale.ENGLISH,
                    "%s,%d,%d%n",
                    estimatedProcedure.name(),
                    estimation.bytesMin,
                    estimation.bytesMax
                );
            }
        } else {
            var mapper = new ObjectMapper();
            // Primary target consumes this from Python, snake_case is more pythonic.
            mapper.setPropertyNamingStrategy(PropertyNamingStrategy.SNAKE_CASE);
            // Pretty print output is nicer to read and any possible overhead from having to parse
            // some whitespace is not important to our use-case
            mapper.enable(SerializationFeature.INDENT_OUTPUT);
            mapper.enable(SerializationFeature.WRITE_SINGLE_ELEM_ARRAYS_UNWRAPPED);

            var jsons = estimatedProcedures
                .stream()
                .map(p -> toJson(p.name(), countOptions, p.estimation()))
                .collect(Collectors.toList());

            mapper.writeValue(System.out, jsons);

        }
    }

    private static JsonOutput toJson(
        String procedureName,
        CountOptions countOptions,
        MemoryEstimateResult memoryEstimation
    ) {
        return ImmutableJsonOutput.builder()
            .bytesMin(memoryEstimation.bytesMin)
            .minMemory(humanReadable(memoryEstimation.bytesMin))
            .bytesMax(memoryEstimation.bytesMax)
            .maxMemory(humanReadable(memoryEstimation.bytesMax))
            .procedure(procedureName)
            .nodeCount(countOptions.nodeCount)
            .relationshipCount(countOptions.relationshipCount)
            .labelCount(countOptions.labelCount)
            .relationshipTypeCount(countOptions.relationshipTypes)
            .nodePropertyCount(countOptions.nodePropertyCount)
            .relationshipPropertyCount(countOptions.relationshipPropertyCount)
            .build();
    }

    @ValueClass
    interface ProcedureMethod {
        String name();

        Method method();
    }

    @ValueClass
    interface EstimatedProcedure {
        String name();

        MemoryEstimateResult estimation();
    }

    @JsonSerialize
    @ValueClass
    interface JsonOutput {
        long bytesMin();

        long bytesMax();

        String minMemory();

        String maxMemory();

        String procedure();

        long nodeCount();

        long relationshipCount();

        int labelCount();

        int relationshipTypeCount();

        int nodePropertyCount();

        int relationshipPropertyCount();
    }
}
