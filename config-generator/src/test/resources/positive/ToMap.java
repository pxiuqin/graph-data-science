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

import org.neo4j.graphalgo.annotation.Configuration;

import java.util.Collection;
import java.util.Map;
import java.util.Optional;

@Configuration("ToMapConfig")
public interface ToMap {

    public static String add42(double d) {
        return String.format("%d42", d);
    }

    @Configuration.Parameter
    int foo();

    long bar();

    @Configuration.ToMapValue("positive.ToMap.add42")
    double baz();

    Optional<Long> maybeBar();

    @Configuration.ToMapValue("positive.ToMap.add42")
    Optional<Double> maybeBaz();

    @Configuration.ToMap
    Map<String, Object> toMap();
}
