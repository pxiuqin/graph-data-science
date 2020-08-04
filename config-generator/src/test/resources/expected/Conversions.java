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
package positive;

import java.util.ArrayList;
import java.util.stream.Collectors;
import javax.annotation.processing.Generated;

import org.jetbrains.annotations.NotNull;
import org.neo4j.graphalgo.core.CypherMapWrapper;

@Generated("org.neo4j.graphalgo.proc.ConfigurationProcessor")
public final class ConversionsConfig implements Conversions.MyConversion {
    private int directMethod;

    private int inheritedMethod;

    private int qualifiedMethod;

    private String referenceTypeAsResult;

    public ConversionsConfig(@NotNull CypherMapWrapper config) {
        ArrayList<IllegalArgumentException> errors = new ArrayList<>();
        try {
            this.directMethod = Conversions.MyConversion.toInt(config.requireString("directMethod"));
        } catch (IllegalArgumentException e) {
            errors.add(e);
        }
        try {
            this.inheritedMethod = Conversions.BaseConversion.toIntBase(config.requireString("inheritedMethod"));
        } catch (IllegalArgumentException e) {
            errors.add(e);
        }
        try {
            this.qualifiedMethod = Conversions.OtherConversion.toIntQual(config.requireString("qualifiedMethod"));
        } catch (IllegalArgumentException e) {
            errors.add(e);
        }
        try {
            this.referenceTypeAsResult = CypherMapWrapper.failOnNull(
                "referenceTypeAsResult",
                Conversions.MyConversion.add42(config.requireString("referenceTypeAsResult"))
            );
        } catch (IllegalArgumentException e) {
            errors.add(e);
        }
        if (!errors.isEmpty()) {
            if (errors.size() == 1) {
                throw errors.get(0);
            } else {
                String combinedErrorMsg = errors
                    .stream()
                    .map(IllegalArgumentException::getMessage)
                    .collect(Collectors.joining(System.lineSeparator() + "\t\t\t\t",
                        "Multiple errors in configuration arguments:" + System.lineSeparator() + "\t\t\t\t",
                        ""
                    ));
                IllegalArgumentException combinedError = new IllegalArgumentException(combinedErrorMsg);
                errors.forEach(error -> combinedError.addSuppressed(error));
                throw combinedError;
            }
        }
    }

    @Override
    public int directMethod() {
        return this.directMethod;
    }

    @Override
    public int inheritedMethod() {
        return this.inheritedMethod;
    }

    @Override
    public int qualifiedMethod() {
        return this.qualifiedMethod;
    }

    @Override
    public String referenceTypeAsResult() {
        return this.referenceTypeAsResult;
    }
}