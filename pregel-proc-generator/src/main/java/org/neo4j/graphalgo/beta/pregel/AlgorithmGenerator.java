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
package org.neo4j.graphalgo.beta.pregel;

import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.CodeBlock;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeSpec;
import org.neo4j.graphalgo.Algorithm;
import org.neo4j.graphalgo.api.Graph;
import org.neo4j.graphalgo.core.concurrency.Pools;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.logging.Log;

import javax.lang.model.SourceVersion;
import javax.lang.model.element.Modifier;
import javax.lang.model.util.Elements;
import java.util.Map;

class AlgorithmGenerator extends PregelGenerator {
    private final PregelValidation.Spec pregelSpec;

    AlgorithmGenerator(Elements elementUtils, SourceVersion sourceVersion, PregelValidation.Spec pregelSpec) {
        super(elementUtils, sourceVersion);
        this.pregelSpec = pregelSpec;
    }

    TypeSpec typeSpec() {
        ClassName algorithmClassName = computationClassName(pregelSpec, ALGORITHM_SUFFIX);

        var typeSpecBuilder = TypeSpec
            .classBuilder(algorithmClassName)
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .superclass(ParameterizedTypeName.get(
                ClassName.get(Algorithm.class),
                algorithmClassName,
                ClassName.get(Pregel.PregelResult.class)
            ))
            .addOriginatingElement(pregelSpec.element());

        addGeneratedAnnotation(typeSpecBuilder);

        typeSpecBuilder.addField(pregelJobField());
        typeSpecBuilder.addMethod(constructor());
        typeSpecBuilder.addMethod(computeMethod());
        typeSpecBuilder.addMethod(meMethod(algorithmClassName));
        typeSpecBuilder.addMethod(releaseMethod());

        return typeSpecBuilder.build();
    }

    private FieldSpec pregelJobField() {
        return FieldSpec
            .builder(
                ParameterizedTypeName.get(
                    ClassName.get(org.neo4j.graphalgo.beta.pregel.Pregel.class),
                    pregelSpec.configTypeName()
                ),
                "pregelJob",
                Modifier.PRIVATE,
                Modifier.FINAL
            )
            .build();
    }

    private MethodSpec constructor() {
        return MethodSpec.constructorBuilder()
            .addParameter(Graph.class, "graph")
            .addParameter(pregelSpec.configTypeName(), "configuration")
            .addParameter(AllocationTracker.class, "tracker")
            .addParameter(Log.class, "log")
            .addStatement(
                CodeBlock.builder().addNamed(
                    "this.pregelJob = $pregel:T.create(" +
                    "graph, " +
                    "configuration, " +
                    "new $computation:T(), " +
                    "$pools:T.DEFAULT," +
                    "tracker" +
                    ")",
                    Map.of(
                        "pregel", Pregel.class,
                        "computation", computationClassName(pregelSpec, ""),
                        "pools", Pools.class
                    )
                )
                    .build()
            )
            .build();
    }

    private MethodSpec computeMethod() {
        return MethodSpec.methodBuilder("compute")
            .addAnnotation(Override.class)
            .addModifiers(Modifier.PUBLIC)
            .returns(Pregel.PregelResult.class)
            .addStatement("return pregelJob.run()")
            .build();
    }

    private MethodSpec releaseMethod() {
        return MethodSpec.methodBuilder("release")
            .addAnnotation(Override.class)
            .addModifiers(Modifier.PUBLIC)
            .addStatement("pregelJob.release()")
            .build();
    }

    private MethodSpec meMethod(ClassName algorithmClassName) {
        return MethodSpec.methodBuilder("me")
            .addAnnotation(Override.class)
            .addModifiers(Modifier.PUBLIC)
            .returns(algorithmClassName)
            .addStatement("return this")
            .build();
    }
}
