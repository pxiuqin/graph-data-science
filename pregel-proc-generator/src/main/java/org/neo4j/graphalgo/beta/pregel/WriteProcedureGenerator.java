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
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import org.neo4j.graphalgo.AlgoBaseProc;
import org.neo4j.graphalgo.WriteProc;
import org.neo4j.graphalgo.beta.pregel.annotation.GDSMode;
import org.neo4j.graphalgo.result.AbstractResultBuilder;

import javax.lang.model.SourceVersion;
import javax.lang.model.element.Modifier;
import javax.lang.model.util.Elements;

class WriteProcedureGenerator extends ProcedureGenerator {

    WriteProcedureGenerator(
        Elements elementUtils,
        SourceVersion sourceVersion,
        PregelValidation.Spec pregelSpec
    ) {
        super(elementUtils, sourceVersion, pregelSpec);
    }

    @Override
    GDSMode procGdsMode() {
        return GDSMode.WRITE;
    }

    @Override
    org.neo4j.procedure.Mode procExecMode() {
        return org.neo4j.procedure.Mode.WRITE;
    }

    @Override
    Class<?> procBaseClass() {
        return WriteProc.class;
    }

    @Override
    Class<?> procResultClass() {
        return PregelWriteResult.class;
    }

    Class<?> procResultBuilderClass() {
        return PregelWriteResult.Builder.class;
    }

    @Override
    MethodSpec procResultMethod() {
        return MethodSpec.methodBuilder("resultBuilder")
            .addAnnotation(Override.class)
            .addModifiers(Modifier.PROTECTED)
            .returns(ParameterizedTypeName.get(AbstractResultBuilder.class, procResultClass()))
            .addParameter(ParameterizedTypeName.get(
                ClassName.get(AlgoBaseProc.ComputationResult.class),
                className(pregelSpec, ALGORITHM_SUFFIX),
                ClassName.get(Pregel.PregelResult.class),
                pregelSpec.configTypeName()
            ), "computeResult")
            .addStatement("var ranIterations = computeResult.result().ranIterations()")
            .addStatement("var didConverge = computeResult.result().didConverge()")
            .addStatement("return new $T().withRanIterations(ranIterations).didConverge(didConverge)", procResultBuilderClass())
            .build();
    }
}
