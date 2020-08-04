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
package org.neo4j.graphalgo.proc;

import com.google.auto.common.GeneratedAnnotationSpecs;
import com.squareup.javapoet.AnnotationSpec;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.CodeBlock;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.NameAllocator;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.neo4j.graphalgo.annotation.Configuration;
import org.neo4j.graphalgo.annotation.Configuration.ConvertWith;
import org.neo4j.graphalgo.annotation.Configuration.Parameter;
import org.neo4j.graphalgo.annotation.ValueClass;
import org.neo4j.graphalgo.core.CypherMapWrapper;

import javax.annotation.processing.Messager;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.google.auto.common.MoreElements.asType;
import static com.google.auto.common.MoreElements.getAnnotationMirror;
import static com.google.auto.common.MoreTypes.asTypeElement;
import static com.google.auto.common.MoreTypes.isTypeOf;
import static javax.lang.model.type.TypeKind.DECLARED;
import static javax.lang.model.util.ElementFilter.methodsIn;

final class GenerateConfiguration {

    private static final String CONFIG_VAR = "config";
    private static final String INSTANCE_VAR = "instance";
    private static final AnnotationSpec NULLABLE = AnnotationSpec.builder(Nullable.class).build();
    private static final AnnotationSpec NOT_NULL = AnnotationSpec.builder(NotNull.class).build();

    private final Messager messager;
    private final Elements elementUtils;
    private final Types typeUtils;
    private final SourceVersion sourceVersion;

    GenerateConfiguration(Messager messager, Elements elementUtils, Types typeUtils, SourceVersion sourceVersion) {
        this.messager = messager;
        this.elementUtils = elementUtils;
        this.typeUtils = typeUtils;
        this.sourceVersion = sourceVersion;
    }

    JavaFile generateConfig(ConfigParser.Spec config, String className) {
        PackageElement rootPackage = elementUtils.getPackageOf(config.root());
        String packageName = rootPackage.getQualifiedName().toString();
        TypeSpec typeSpec = process(config, packageName, className);
        return JavaFile
            .builder(packageName, typeSpec)
            .indent("    ")
            .skipJavaLangImports(true)
            .build();
    }

    private TypeSpec process(ConfigParser.Spec config, String packageName, String generatedClassName) {
        TypeSpec.Builder builder = classBuilder(config, packageName, generatedClassName);

        FieldDefinitions fieldDefinitions = defineFields(config);
        builder.addFields(fieldDefinitions.fields());

        MethodSpec constructor = defineConstructor(config, fieldDefinitions.names());
        Optional<MethodSpec> factory = defineFactory(
            config,
            generatedClassName,
            constructor,
            fieldDefinitions.names()
        );
        if (factory.isPresent()) {
            MethodSpec privateConstructor = MethodSpec.constructorBuilder()
                .addAnnotations(constructor.annotations)
                .addParameters(constructor.parameters)
                .addCode(constructor.code)
                .addModifiers(Modifier.PRIVATE)
                .build();
            builder.addMethod(privateConstructor);
            builder.addMethod(factory.get());
        } else {
            builder.addMethod(constructor);
        }

        return builder
            .addMethods(defineGetters(config, fieldDefinitions.names()))
            .build();
    }

    private TypeSpec.Builder classBuilder(ConfigParser.Spec config, String packageName, String generatedClassName) {
        TypeSpec.Builder classBuilder = createNewClass(config, packageName, generatedClassName)
            .addSuperinterface(TypeName.get(config.rootType()));
        addGeneratedAnnotation(classBuilder);
        return classBuilder;
    }

    private TypeSpec.Builder createNewClass(ConfigParser.Spec config, String packageName, String generatedClassName) {
        return TypeSpec
            .classBuilder(ClassName.get(packageName, generatedClassName))
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addOriginatingElement(config.root());
    }

    private void addGeneratedAnnotation(TypeSpec.Builder classBuilder) {
        GeneratedAnnotationSpecs.generatedAnnotationSpec(
            elementUtils,
            sourceVersion,
            ConfigurationProcessor.class
        ).ifPresent(classBuilder::addAnnotation);
    }

    private FieldDefinitions defineFields(ConfigParser.Spec config) {
        NameAllocator names = new NameAllocator();
        ImmutableFieldDefinitions.Builder builder = ImmutableFieldDefinitions.builder().names(names);
        config.members().stream().filter(ConfigParser.Member::isConfigValue).map(member ->
            FieldSpec.builder(
                member.typeSpecWithAnnotation(Nullable.class),
                names.newName(member.methodName(), member),
                Modifier.PRIVATE
            ).build()
        ).forEach(builder::addField);
        return builder.build();
    }

    private MethodSpec defineConstructor(ConfigParser.Spec config, NameAllocator names) {
        MethodSpec.Builder configMapConstructor = MethodSpec
            .constructorBuilder()
            .addModifiers(Modifier.PUBLIC);

        String configParameterName = names.newName(CONFIG_VAR, CONFIG_VAR);
        boolean requiredMapParameter = false;

        String errorsVarName = names.newName("errors");
        if (!config.members().isEmpty()) {
            configMapConstructor.addStatement("$1T<$2T> $3N = new $1T<>()", ArrayList.class, IllegalArgumentException.class, errorsVarName);
        }

        for (ConfigParser.Member member : config.members()) {
            Optional<MemberDefinition> memberDefinition = memberDefinition(names, member);
            if (memberDefinition.isPresent()) {
                ExecutableElement method = member.method();
                MemberDefinition definition = memberDefinition.get();

                Parameter parameter = method.getAnnotation(Parameter.class);
                if (parameter == null) {
                    requiredMapParameter = true;
                    addConfigGetterToConstructor(
                        configMapConstructor,
                        definition,
                        errorsVarName
                    );
                } else {
                    addParameterToConstructor(
                        configMapConstructor,
                        definition,
                        parameter,
                        errorsVarName
                    );
                }
            }
        }

        for (ConfigParser.Member member : config.members()) {
            if (member.validates()) {
                catchValidationError(configMapConstructor, errorsVarName, (builder) -> builder.addStatement("$N()", member.methodName()));
            }
        }

        if (!config.members().isEmpty()) {
            combineCollectedErrors(names, configMapConstructor, errorsVarName);
        }

        if (requiredMapParameter) {
            configMapConstructor.addParameter(
                TypeName.get(CypherMapWrapper.class).annotated(NOT_NULL),
                configParameterName
            );
        }

        return configMapConstructor.build();
    }

    private void combineCollectedErrors(
        NameAllocator names,
        MethodSpec.Builder configMapConstructor,
        String errorsVarName
    ) {
        String combinedErrorMsgVarName = names.newName("combinedErrorMsg");
        String combinedErrorVarName = names.newName("combinedError");
        configMapConstructor.beginControlFlow("if(!$N.isEmpty())", errorsVarName)
            .beginControlFlow("if($N.size() == $L)", errorsVarName, 1)
            .addStatement("throw $N.get($L)", errorsVarName, 0)
            .nextControlFlow("else")
            .addStatement(
                "$1T $2N = $3N.stream().map($4T::getMessage)" +
                ".collect($5T.joining(System.lineSeparator() + $6S, $7S + System.lineSeparator() + $6S, $8S))",
                String.class,
                combinedErrorMsgVarName,
                errorsVarName,
                IllegalArgumentException.class,
                Collectors.class,
                "\t\t\t\t",
                "Multiple errors in configuration arguments:", //prefix
                "" // suffix
            )
            .addStatement(
                "$1T $2N = new $1T($3N)",
                IllegalArgumentException.class,
                combinedErrorVarName,
                combinedErrorMsgVarName
            )
            .addStatement(
                "$1N.forEach($2N -> $3N.addSuppressed($2N))",
                errorsVarName,
                names.newName("error"),
                combinedErrorVarName
            )
            .addStatement("throw $N", combinedErrorVarName)
            .endControlFlow()
            .endControlFlow();
    }

    private void catchAndPropagateIllegalArgumentError(
        MethodSpec.Builder builder,
        String errorVarName,
        UnaryOperator<MethodSpec.Builder> statementFunc
    ) {
        builder.beginControlFlow("try");
        statementFunc.apply(builder);
        builder
            .nextControlFlow("catch ($T e)", IllegalArgumentException.class)
            .addStatement("$N.add(e)", errorVarName)
            .endControlFlow();
    }

    private void catchValidationError(
        MethodSpec.Builder builder,
        String errorVarName,
        UnaryOperator<MethodSpec.Builder> statementFunc
    ) {
        builder.beginControlFlow("try");
        statementFunc.apply(builder);
        builder
            .nextControlFlow("catch ($T e)", IllegalArgumentException.class)
            .addStatement("$N.add(e)", errorVarName)
            // should only throw NPE if previously an error occured on the field it valides on (field is null then)
            .nextControlFlow("catch ($T e)", NullPointerException.class)
            .endControlFlow();
    }

    private Optional<MethodSpec> defineFactory(
        ConfigParser.Spec config,
        String generatedClassName,
        MethodSpec constructor,
        NameAllocator names
    ) {
        List<ConfigParser.Member> normalizers = config
            .members()
            .stream()
            .filter(ConfigParser.Member::normalizes)
            .collect(Collectors.toList());

        if (normalizers.isEmpty()) {
            return Optional.empty();
        }

        CodeBlock constructorArgs = constructor.parameters
            .stream()
            .map(param -> CodeBlock.of("$N", param))
            .collect(CodeBlock.joining(", "));

        String instanceVarName = names.newName(INSTANCE_VAR, INSTANCE_VAR);
        TypeName interfaceType = TypeName.get(config.rootType());

        MethodSpec.Builder factory = MethodSpec
            .methodBuilder("of")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(interfaceType)
            .addParameters(constructor.parameters)
            .addStatement("$T $N = new $L($L)", interfaceType, instanceVarName, generatedClassName, constructorArgs);

        for (ConfigParser.Member member : normalizers) {
            factory.addStatement("$1N = $1N.$2N()", instanceVarName, member.methodName());
        }

        return Optional.of(factory
            .addStatement("return $N", instanceVarName)
            .build());
    }

    private void addConfigGetterToConstructor(
        MethodSpec.Builder constructor,
        MemberDefinition definition,
        String errorsVarName
    ) {
        CodeBlock.Builder code = CodeBlock.builder().add(
            "$N.$L$L($S",
            definition.configParamName(),
            definition.methodPrefix(),
            definition.methodName(),
            definition.configKey()
        );
        definition.defaultProvider().ifPresent(d -> code.add(", $L", d));
        definition.expectedType().ifPresent(t -> code.add(", $L", t));
        CodeBlock codeBlock = code.add(")").build();
        for (UnaryOperator<CodeBlock> converter : definition.converters()) {
            codeBlock = converter.apply(codeBlock);
        }
        TypeMirror fieldType = definition.fieldType();
        if (fieldType.getKind() == DECLARED) {
            boolean isNullable = !definition.member().annotations(Nullable.class).isEmpty();

            if (!isNullable) {
                codeBlock = CodeBlock.of(
                    "$T.failOnNull($S, $L)",
                    CypherMapWrapper.class,
                    definition.configKey(),
                    codeBlock
                );
            }
        }

        if (definition.member().validatesIntegerRange()) {
            Configuration.IntegerRange range = definition
                .member()
                .method()
                .getAnnotation(Configuration.IntegerRange.class);
            codeBlock = CodeBlock.of(
                "$T.validateIntegerRange($S, $L, $L, $L, $L, $L)",
                CypherMapWrapper.class,
                definition.configKey(),
                codeBlock,
                elementUtils.getConstantExpression(range.min()),
                elementUtils.getConstantExpression(range.max()),
                elementUtils.getConstantExpression(range.minInclusive()),
                elementUtils.getConstantExpression(range.maxInclusive())
            );
        }

        if (definition.member().validatesDoubleRange()) {
            Configuration.DoubleRange range = definition
                .member()
                .method()
                .getAnnotation(Configuration.DoubleRange.class);
            codeBlock = CodeBlock.of(
                "$T.validateDoubleRange($S, $L, $L, $L, $L, $L)",
                CypherMapWrapper.class,
                definition.configKey(),
                codeBlock,
                elementUtils.getConstantExpression(range.min()),
                elementUtils.getConstantExpression(range.max()),
                elementUtils.getConstantExpression(range.minInclusive()),
                elementUtils.getConstantExpression(range.maxInclusive())
            );
        }

        CodeBlock finalCodeBlock = codeBlock;
        catchAndPropagateIllegalArgumentError(
            constructor,
            errorsVarName,
            (builder) -> builder.addStatement("this.$N = $L", definition.fieldName(), finalCodeBlock)
        );
    }

    private void addParameterToConstructor(
        MethodSpec.Builder constructor,
        MemberDefinition definition,
        Parameter parameter,
        String errorsVarName
    ) {
        TypeName paramType = TypeName.get(definition.parameterType());

        CodeBlock valueProducer;
        if (definition.parameterType().getKind() == DECLARED) {
            if (parameter.acceptNull()) {
                paramType = paramType.annotated(NULLABLE);
                valueProducer = CodeBlock.of("$N", definition.fieldName());
            } else {
                paramType = paramType.annotated(NOT_NULL);
                valueProducer = CodeBlock.of(
                    "$T.failOnNull($S, $N)",
                    CypherMapWrapper.class,
                    definition.configKey(),
                    definition.fieldName()
                );
            }
        } else {
            valueProducer = CodeBlock.of("$N", definition.fieldName());
        }

        for (UnaryOperator<CodeBlock> converter : definition.converters()) {
            valueProducer = converter.apply(valueProducer);
        }

        CodeBlock finalValueProducer = valueProducer;
        constructor.addParameter(paramType, definition.fieldName());
        catchAndPropagateIllegalArgumentError(
            constructor,
            errorsVarName,
            (builder) -> builder.addStatement("this.$N = $L", definition.fieldName(), finalValueProducer)
        );
    }

    private Optional<MemberDefinition> memberDefinition(NameAllocator names, ConfigParser.Member member) {
        if (!member.isConfigValue()) {
            return Optional.empty();
        }

        ExecutableElement method = member.method();
        TypeMirror targetType = method.getReturnType();
        ConvertWith convertWith = method.getAnnotation(ConvertWith.class);
        if (convertWith == null) {
            return memberDefinition(names, member, targetType);
        }

        String converter = convertWith.value().trim();
        if (converter.isEmpty()) {
            return converterError(method, "Empty conversion method is not allowed");
        }

        if (!converter.contains("#")) {
            return memberDefinition(
                names,
                member,
                targetType,
                asType(method.getEnclosingElement()),
                converter,
                true
            );
        }

        String[] nameParts = converter.split(Pattern.quote("#"), 2);
        String methodName = nameParts[1];
        if (methodName.isEmpty() || methodName.contains("#")) {
            return converterError(
                method,
                "[%s] is not a valid fully qualified method name: " +
                "it must start with a fully qualified class name followed by a '#' " +
                "and then the method name",
                converter
            );
        }

        String className = nameParts[0];
        TypeElement classElement = elementUtils.getTypeElement(className);
        if (classElement == null) {
            return converterError(
                method,
                "[%s] is not a valid fully qualified method name: " +
                "The class [%s] cannot be found",
                converter,
                className
            );
        }

        return memberDefinition(names, member, targetType, classElement, methodName, false);
    }

    private Optional<MemberDefinition> memberDefinition(
        NameAllocator names,
        ConfigParser.Member member,
        TypeMirror targetType,
        TypeElement classElement,
        CharSequence methodName,
        boolean scanInheritance
    ) {
        String converter = member.method().getAnnotation(ConvertWith.class).value();
        List<ExecutableElement> validCandidates = new ArrayList<>();
        Collection<InvalidCandidate> invalidCandidates = new ArrayList<>();
        Deque<TypeElement> classesToSearch = new ArrayDeque<>();
        classesToSearch.addLast(classElement);
        do {
            TypeElement currentClass = classesToSearch.pollFirst();
            if (currentClass == null) {
                return converterError(classElement, "Inherited interface was null, this could be a bug in the JDK.");
            }

            validCandidates.clear();

            for (ExecutableElement candidate : methodsIn(currentClass.getEnclosedElements())) {
                if (!candidate.getSimpleName().contentEquals(methodName)) {
                    continue;
                }

                int sizeBeforeValidation = invalidCandidates.size();

                Set<Modifier> modifiers = candidate.getModifiers();
                if (!modifiers.contains(Modifier.STATIC)) {
                    invalidCandidates.add(InvalidCandidate.of(candidate, "Must be static"));
                }
                if (!modifiers.contains(Modifier.PUBLIC)) {
                    invalidCandidates.add(InvalidCandidate.of(candidate, "Must be public"));
                }
                if (!candidate.getTypeParameters().isEmpty()) {
                    invalidCandidates.add(InvalidCandidate.of(candidate, "May not be generic"));
                }
                if (!candidate.getThrownTypes().isEmpty()) {
                    invalidCandidates.add(InvalidCandidate.of(candidate, "May not declare any exceptions"));
                }
                if (!(candidate.getParameters().size() == 1)) {
                    invalidCandidates.add(InvalidCandidate.of(candidate, "May only accept one parameter"));
                }
                if (!typeUtils.isAssignable(candidate.getReturnType(), targetType)) {
                    invalidCandidates.add(InvalidCandidate.of(
                        candidate,
                        "Must return a type that is assignable to %s",
                        targetType
                    ));
                }

                if (invalidCandidates.size() == sizeBeforeValidation) {
                    validCandidates.add(candidate);
                }
            }

            if (validCandidates.size() > 1) {
                for (ExecutableElement candidate : validCandidates) {
                    error(
                        String.format(Locale.ENGLISH,"Method is ambiguous and a possible candidate for [%s]", converter),
                        candidate
                    );
                }
                return converterError(
                    member.method(), "Multiple possible candidates found: %s", validCandidates
                );
            }

            if (validCandidates.size() == 1) {
                ExecutableElement candidate = validCandidates.get(0);
                VariableElement parameter = candidate.getParameters().get(0);
                TypeMirror currentType = currentClass.asType();
                return memberDefinition(names, member, parameter.asType())
                    .map(d -> ImmutableMemberDefinition.builder()
                        .from(d)
                        .addConverter(c -> CodeBlock.of(
                            "$T.$N($L)",
                            currentType,
                            candidate.getSimpleName().toString(),
                            c
                        ))
                        .build()
                    );
            }

            if (scanInheritance) {
                for (TypeMirror superInterface : currentClass.getInterfaces()) {
                    classesToSearch.addLast(asTypeElement(superInterface));
                }
            }
        } while (!classesToSearch.isEmpty());

        for (InvalidCandidate invalidCandidate : invalidCandidates) {
            error(String.format(Locale.ENGLISH,invalidCandidate.message(), invalidCandidate.args()), invalidCandidate.element());
        }

        return converterError(
            member.method(),
            "No suitable method found that matches [%s]. " +
            "Make sure that the method is static, public, unary, not generic, " +
            "does not declare any exception and returns [%s]",
            converter,
            targetType
        );
    }

    private Optional<MemberDefinition> memberDefinition(
        NameAllocator names,
        ConfigParser.Member member,
        TypeMirror targetType
    ) {
        ImmutableMemberDefinition.Builder builder = ImmutableMemberDefinition
            .builder()
            .member(member)
            .fieldName(names.get(member))
            .configParamName(names.get(CONFIG_VAR))
            .configKey(member.lookupKey())
            .fieldType(member.method().getReturnType())
            .parameterType(targetType)
            .methodPrefix("require");

        switch (targetType.getKind()) {
            case BOOLEAN:
                builder.methodName("Bool");
                break;
            case INT:
                builder.methodName("Int");
                break;
            case LONG:
                builder.methodName("Long");
                break;
            case DOUBLE:
                builder.methodName("Double");
                break;
            case BYTE:
            case SHORT:
            case FLOAT:
                builder
                    .methodName("Number")
                    .addConverter(c -> CodeBlock.of("$L.$LValue()", c, targetType));
                break;
            case DECLARED:
                if (isTypeOf(String.class, targetType)) {
                    builder.methodName("String");
                } else if (isTypeOf(Number.class, targetType)) {
                    builder.methodName("Number");
                } else if (isTypeOf(Optional.class, targetType)) {
                    if (member.method().isDefault()) {
                        return error(
                            "Optional fields can not to be declared default (Optional.empty is the default).",
                            member.method()
                        );
                    }
                    List<? extends TypeMirror> typeArguments = ((DeclaredType) targetType).getTypeArguments();
                    if (typeArguments.isEmpty()) {
                        return error(
                            "Optional must have a Cypher-supported type as type argument, but found none.",
                            member.method()
                        );
                    }
                    builder
                        .methodPrefix("get")
                        .methodName("Optional")
                        .expectedType(CodeBlock.of("$T.class", ClassName.get(asTypeElement(typeArguments.get(0)))));
                } else {
                    builder
                        .methodName("Checked")
                        .expectedType(CodeBlock.of("$T.class", ClassName.get(asTypeElement(targetType))));
                }
                break;
            default:
                return error("Unsupported return type: " + targetType, member.method());
        }

        if (member.method().isDefault()) {
            builder
                .methodPrefix("get")
                .defaultProvider(CodeBlock.of(
                    "$T.super.$N()",
                    member.owner().asType(),
                    member.methodName()
                    )
                );
        }

        return Optional.of(builder.build());
    }

    private void injectToMapCode(ConfigParser.Spec config, MethodSpec.Builder builder) {
        List<ConfigParser.Member> configMembers = config
            .members()
            .stream()
            .filter(ConfigParser.Member::isMapParameter)
            .collect(Collectors.toList());

        switch (configMembers.size()) {
            case 0:
                builder.addStatement("return $T.emptyMap()", Collections.class);
                break;
            case 1:
                ConfigParser.Member singleConfigMember = configMembers.iterator().next();
                String parameter = singleConfigMember.lookupKey();
                builder.addStatement(
                    "return $T.singletonMap($S, $L)",
                    Collections.class,
                    parameter,
                    getMapValueCode(singleConfigMember)
                );
                break;
            default:
                builder.addStatement("$T<$T, Object> map = new $T<>()", Map.class, String.class, LinkedHashMap.class);
                configMembers.forEach(configMember -> {
                    if (isTypeOf(Optional.class, configMember.method().getReturnType())) {
                        builder.addStatement(getMapPutOptionalCode(configMember));
                    } else {
                        builder.addStatement(
                            "map.put($S, $L)",
                            configMember.lookupKey(),
                            getMapValueCode(configMember)
                        );
                    }
                });
                builder.addStatement("return map");
                break;
        }
    }

    @NotNull
    private CodeBlock getMapValueCode(ConfigParser.Member configMember) {
        String getter = configMember.methodName();
        Configuration.ToMapValue toMapValue = configMember.method().getAnnotation(Configuration.ToMapValue.class);
        return (toMapValue == null)
            ? CodeBlock.of("$N()", getter)
            : CodeBlock.of("$L($N())", getReference(toMapValue), getter);
    }

    @NotNull
    private CodeBlock getMapPutOptionalCode(ConfigParser.Member configMember) {
        Configuration.ToMapValue toMapValue = configMember.method().getAnnotation(Configuration.ToMapValue.class);

        CodeBlock mapValue = (toMapValue == null)
            ? CodeBlock.of("$L", configMember.lookupKey())
            : CodeBlock.of("$L($L)", getReference(toMapValue), configMember.lookupKey());

        return CodeBlock.of("$L.ifPresent($L -> map.put($S, $L))",
            CodeBlock.of("$N()", configMember.methodName()),
            configMember.lookupKey(),
            configMember.lookupKey(),
            mapValue
        );
    }

    private String getReference(Configuration.ToMapValue toMapValue) {
        return toMapValue.value().replaceAll("#", ".");
    }

    private CodeBlock collectKeysCode(ConfigParser.Spec config) {
        Collection<String> configKeys = config
            .members()
            .stream()
            .filter(ConfigParser.Member::isMapParameter)
            .map(ConfigParser.Member::lookupKey)
            .collect(Collectors.toCollection(LinkedHashSet::new));

        switch (configKeys.size()) {
            case 0:
                return CodeBlock.of("return $T.emptyList()", Collections.class);
            case 1:
                return CodeBlock.of("return $T.singleton($S)", Collections.class, configKeys.iterator().next());
            default:
                CodeBlock keys = configKeys
                    .stream()
                    .map(name -> CodeBlock.of("$S", name))
                    .collect(CodeBlock.joining(", "));
                return CodeBlock.of("return $T.asList($L)", Arrays.class, keys);
        }
    }

    private Iterable<MethodSpec> defineGetters(ConfigParser.Spec config, NameAllocator names) {
        return config
            .members()
            .stream()
            .map(member -> defineGetter(config, names, member))
            .flatMap(o -> o.map(Stream::of).orElseGet(Stream::empty))
            .collect(Collectors.toList());
    }

    private Optional<MethodSpec> defineGetter(
        ConfigParser.Spec config,
        NameAllocator names,
        ConfigParser.Member member
    ) {
        if (member.isConfigValue() || member.collectsKeys() || member.toMap()) {
            MethodSpec.Builder builder = MethodSpec
                .overriding(member.method())
                .returns(member.typeSpecWithAnnotation(Nullable.class));
            if (member.collectsKeys()) {
                builder.addStatement(collectKeysCode(config));
            } else if (member.toMap()) {
                injectToMapCode(config, builder);
            }
            else {
                builder.addStatement("return this.$N", names.get(member));
            }
            return Optional.of(builder.build());
        }
        return Optional.empty();
    }

    private <T> Optional<T> error(CharSequence message, Element element) {
        messager.printMessage(
            Diagnostic.Kind.ERROR,
            message,
            element
        );
        return Optional.empty();
    }

    private <T> Optional<T> converterError(Element element, String message, Object... args) {
        messager.printMessage(
            Diagnostic.Kind.ERROR,
            String.format(Locale.ENGLISH, message, args),
            element,
            getAnnotationMirror(element, ConvertWith.class).orNull()
        );
        return Optional.empty();
    }

    @ValueClass
    interface FieldDefinitions {
        List<FieldSpec> fields();

        NameAllocator names();
    }

    @ValueClass
    interface MemberDefinition {
        ConfigParser.Member member();

        TypeMirror fieldType();

        TypeMirror parameterType();

        String fieldName();

        String configParamName();

        String methodPrefix();

        String methodName();

        String configKey();

        Optional<CodeBlock> defaultProvider();

        Optional<CodeBlock> expectedType();

        List<UnaryOperator<CodeBlock>> converters();
    }

    @ValueClass
    interface InvalidCandidate {
        Element element();

        String message();

        Object[] args();

        static InvalidCandidate of(Element element, String format, Object... args) {
            return ImmutableInvalidCandidate.builder().element(element).message(format).args(args).build();
        }
    }
}
