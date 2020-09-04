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
package org.neo4j.graphalgo.model.catalog;

import org.neo4j.graphalgo.core.model.Model;
import org.neo4j.graphalgo.core.model.ModelCatalog;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.Collection;
import java.util.stream.Stream;

import static org.neo4j.procedure.Mode.READ;

public class ModelListProc extends ModelCatalogProc {

    private static final String DESCRIPTION = "Lists all models contained in the model catalog.";

    @Procedure(name = "gds.beta.model.list", mode = READ)
    @Description(DESCRIPTION)
    public Stream<ModelResult> list(@Name(value = "modelName", defaultValue = NO_VALUE) String modelName) {
        if (modelName == null || modelName.equals(NO_VALUE)) {
            Collection<Model<?, ?>> models = ModelCatalog.list(username());
            return models.stream().map(ModelResult::new);
        } else {
            validateModelName(modelName);
            return Stream.of(new ModelResult(ModelCatalog.list(username(), modelName)));
        }
    }
}
