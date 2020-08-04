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
package org.neo4j.graphalgo.compat;

import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.kernel.impl.factory.DbmsInfo;

import java.nio.file.Path;

final class CompatGraphDatabaseAPI42 extends GdsGraphDatabaseAPI {

    CompatGraphDatabaseAPI42(DatabaseManagementService dbms) {
        super(dbms);
    }

    @Override
    public Path dbHome(Path workingDir) {
        return api.databaseLayout().getNeo4jLayout().homeDirectory();
    }

    @Override
    public DbmsInfo dbmsInfo() {
        return api.dbmsInfo();
    }
}
