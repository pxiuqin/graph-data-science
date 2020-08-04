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

import org.junit.jupiter.api.Test;
import org.neo4j.graphalgo.config.AlgoBaseConfig;
import org.neo4j.graphalgo.config.BaseConfig;
import org.neo4j.graphalgo.core.CypherMapWrapper;
import org.neo4j.graphalgo.core.loading.GraphStoreCatalog;
import org.neo4j.graphalgo.utils.ExceptionUtil;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphalgo.utils.StringFormatting.formatWithLocale;

public interface HeapControlTest<ALGORITHM extends Algorithm<ALGORITHM, RESULT>, CONFIG extends AlgoBaseConfig, RESULT> extends AlgoBaseProcTest<ALGORITHM, CONFIG, RESULT> {
    String DB_CYPHER = "CREATE " +
                       " (zhen:Person {name: 'Zhen'})," +
                       " (praveena:Person {name: 'Praveena'})," +
                       " (michael:Person {name: 'Michael'})," +
                       " (arya:Person {name: 'Arya'})," +
                       " (karin:Person {name: 'Karin'})," +

                       " (zhen)-[:FRIENDS]->(arya)," +
                       " (zhen)-[:FRIENDS]->(praveena)," +
                       " (praveena)-[:WORKS_WITH]->(karin)," +
                       " (praveena)-[:FRIENDS]->(michael)," +
                       " (michael)-[:WORKS_WITH]->(karin)," +
                       " (arya)-[:FRIENDS]->(karin)";

    @Test
    default void shouldPassOnSufficientMemory() {
        applyOnProcedure(proc -> {
            CONFIG config = proc.newConfig(Optional.empty(), createMinimalImplicitConfig(CypherMapWrapper.empty()));
            proc.tryValidateMemoryUsage(config, proc::memoryEstimation, () -> 10000000);
        });
    }

    @Test
    default void shouldFailOnInsufficientMemory() {
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
            applyOnProcedure(proc -> {
                CONFIG config = proc.newConfig(Optional.empty(), createMinimalImplicitConfig(CypherMapWrapper.empty()));
                proc.tryValidateMemoryUsage(config, proc::memoryEstimation, () -> 42);
            });
        });

        String message = ExceptionUtil.rootCause(exception).getMessage();
        String messageTemplate = "Procedure was blocked since minimum estimated memory \\(.+\\) exceeds current free memory \\(42 Bytes\\).";
        if (GraphStoreCatalog.graphStoresCount() > 0) {
            messageTemplate += formatWithLocale(
                " Note: there are %s graphs currently loaded into memory.",
                GraphStoreCatalog.graphStoresCount()
            );
        }
        System.out.println(message);
        System.out.println(messageTemplate);
        assertTrue(message.matches(
            messageTemplate));
    }

    @Test
    default void shouldNotFailOnInsufficientMemoryIfInSudoMode() {
        applyOnProcedure(proc -> {
            CypherMapWrapper configMap = CypherMapWrapper.empty().withBoolean(BaseConfig.SUDO_KEY, true);
            CONFIG config = proc.newConfig(Optional.empty(), createMinimalImplicitConfig(configMap));
            proc.tryValidateMemoryUsage(config, proc::memoryEstimation, () -> 42);
        });
    }
}
