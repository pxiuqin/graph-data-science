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
package org.neo4j.graphalgo.nodesim;

import org.neo4j.graphalgo.core.loading.HugeGraphUtil;

import java.util.Comparator;
import java.util.Objects;

//相似度结果
public class SimilarityResult implements Comparable<SimilarityResult>, HugeGraphUtil.Relationship {

    static Comparator<SimilarityResult> ASCENDING = SimilarityResult::compareTo;  //正序
    static Comparator<SimilarityResult> DESCENDING = ASCENDING.reversed();    //逆序

    public long node1;
    public long node2;
    public double similarity;

    public SimilarityResult(long node1, long node2, double similarity) {
        this.node1 = node1;
        this.node2 = node2;
        this.similarity = similarity;
    }

    @Override
    public long sourceNodeId() {
        return node1;
    }

    @Override
    public long targetNodeId() {
        return node2;
    }

    @Override
    public double property() {
        return similarity;
    }

    //求相反
    public SimilarityResult reverse() {
        return new SimilarityResult(node2, node1, similarity);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SimilarityResult that = (SimilarityResult) o;
        return node1 == that.node1 &&
               node2 == that.node2 &&
               Double.compare(that.similarity, similarity) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(node1, node2, similarity);
    }

    @Override
    public String toString() {
        return "SimilarityResult{" +
               "node1=" + node1 +
               ", node2=" + node2 +
               ", similarity=" + similarity +
               '}';
    }

    @Override
    public int compareTo(SimilarityResult o) {
        return Double.compare(this.similarity, o.similarity);
    }
}
