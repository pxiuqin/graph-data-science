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
package org.neo4j.graphalgo.core.utils.paged.dss;

import org.neo4j.graphalgo.api.NodeProperties;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;
import org.neo4j.graphalgo.core.utils.mem.AllocationTracker;
import org.neo4j.graphalgo.core.utils.paged.HugeAtomicLongArray;
import org.neo4j.graphalgo.core.utils.paged.LongPageCreator;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Add adaption of the C++ implementation [1] for the
 * "Wait-free Parallel Algorithms for the Union-Find Problem" [2]
 * with some input from an atomic DSS implementation in Rust [3].
 *
 * The major difference for our DSS is, that we don't supported the
 * Union-by-Rank strategy [3], for technical and performance reasons.
 *
 * The reference implementation in C++ uses 32bit unsigned integers for
 * both the id values and the rank values. Those two values have to be
 * updated atomically, which [1] does by merging them into a single
 * 64bit unsigned integer and doing atomic/cas operations on that value.
 *
 * We need 64bits for the id value alone and since there is no u128 data type
 * in Java, the only way to update those values would be to use a class for
 * the combination of id+rank and updated the references to that atomically.
 * This is the approach that the Rust implementation is doing, except that
 * Rust allows the struct values to be allocated on the stack and has no GC
 * overhead, where that would not be true for Java (in the near future).
 *
 * We drop the by-Rank functionality and just support Union-by-Min for this DSS.
 *
 * The main difference in implementation compared to the regular DSS is that we
 * use CAS operations to atomically set a set id for some value.
 * We will retry union operations until a thread succeeds in changing the set id
 * for a node. Other threads that might have wanted to write a different value
 * will fail and the CAS operation and redo their union step. This allows for concurrent
 * writes into a single DSS and does not longer necessitate an additional merge step.
 *
 * <ul>
 * <li>[1]: <a href="https://github.com/wjakob/dset/blob/7967ef0e6041cd9d73b9c7f614ab8ae92e9e587a/dset.h">{@code https://github.com/wjakob/dset/blob/7967ef0e6041cd9d73b9c7f614ab8ae92e9e587a/dset.h}</a></li>
 * <li>[2]: <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.8354&amp;rep=rep1&amp;type=pdf">{@code http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.8354&rep=rep1&type=pdf}</a></li>
 * <li>[3]: <a href="https://github.com/tov/disjoint-sets-rs/blob/88ab08df21f04fcf7c157b6e042efd561ee873ba/src/concurrent.rs">{@code https://github.com/tov/disjoint-sets-rs/blob/88ab08df21f04fcf7c157b6e042efd561ee873ba/src/concurrent.rs}</a></li>
 * <li>[4]: <a href="https://en.wikipedia.org/wiki/Disjoint-set_data_structure#by_rank">{@code https://en.wikipedia.org/wiki/Disjoint-set_data_structure#by_rank}</a></li>
 * </ul>
 *
 * 1.makeSet(s)：建立一个新的并查集，其中包含 s 个单元素集合。
 * 2.unionSet(x, y)：把元素 x 和元素 y 所在的集合合并，要求 x 和 y 所在的集合不相交，如果相交则不合并。
 * 3.find(x)：找到元素 x 所在的集合的代表，该操作也可以用于判断两个元素是否位于同一个集合，只要将它们各自的代表比较一下就可以了。
 * https://www.cnblogs.com/cyjb/p/UnionFindSets.html
 */
public final class HugeAtomicDisjointSetStruct implements DisjointSetStruct {

    private static final int NO_SUCH_SEED_VALUE = 0;

    public static MemoryEstimation memoryEstimation(boolean incremental) {
        MemoryEstimations.Builder builder = MemoryEstimations
                .builder(HugeAtomicDisjointSetStruct.class)
                .perNode("data", HugeAtomicLongArray::memoryEstimation);
        if (incremental) {
            builder.perNode("seeding information", HugeAtomicLongArray::memoryEstimation);
        }
        return builder.build();
    }

    private final HugeAtomicLongArray parent;
    private final HugeAtomicLongArray communities;
    private final AtomicLong maxCommunityId;

    public HugeAtomicDisjointSetStruct(long capacity, AllocationTracker tracker, int concurrency) {
        this.parent = HugeAtomicLongArray.newArray(capacity, LongPageCreator.identity(concurrency), tracker);
        this.communities = null;
        this.maxCommunityId = null;
    }

    public HugeAtomicDisjointSetStruct(
        long capacity,
        NodeProperties communityMapping,
        AllocationTracker tracker,
        int concurrency
    ) {
        this.parent = HugeAtomicLongArray.newArray(capacity, LongPageCreator.identity(concurrency), tracker);
        this.communities = HugeAtomicLongArray.newArray(
            capacity,
            LongPageCreator.of(concurrency, nodeId -> {
                var seedCommunity = communityMapping.longValue(nodeId);
                return seedCommunity < 0 ? -1 : seedCommunity;
            }),
            tracker
        );
        maxCommunityId = new AtomicLong(communityMapping.getMaxLongPropertyValue().orElse(NO_SUCH_SEED_VALUE));
    }

    private long parent(long id) {
        return parent.get(id);
    }

    private long find(long id) {
        long parent;
        while (id != (parent = parent(id))) {
            long grandParent = parent(parent);
            if (parent != grandParent) {
                // Try to apply path-halving by setting the value
                // for some id to its grand parent. This might fail
                // if another thread is also changing the same value
                // but that's ok. The CAS operations guarantees
                // that at least one of the contenting threads will
                // succeed. That's enough for the path-halving to work
                // and there is no need to retry in case of a CAS failure.
                this.parent.compareAndSet(id, parent, grandParent);
            }
            id = grandParent;
        }
        return id;
    }

    @Override
    public long setIdOf(final long nodeId) {
        long setId = find(nodeId);
        if (communities == null) {
            return setId;  //直接返回当前ID
        }

        do {
            long providedSetId = communities.get(setId);
            if (providedSetId >= 0L) {
                return providedSetId;    //返回记录的community_id
            }
            long newSetId = maxCommunityId.incrementAndGet();   //community_id 累加
            if (communities.compareAndSet(setId, providedSetId, newSetId)) {
                return newSetId;  //返回新的community_id
            }
        } while (true);
    }

    @Override
    public boolean sameSet(long id1, long id2) {
        while (true) {
            id1 = find(id1);
            id2 = find(id2);
            if (id1 == id2) {
                return true;  //community_id相同
            }
            if (parent(id1) == id1) {  //不在同一集合
                return false;
            }
        }
    }

    //这里也可以应用一个简单的启发式策略——按秩合并。该方法使用秩来表示树高度的上界，在合并时，总是将具有较小秩的树根指向具有较大秩的树根。
    //简单的说，就是总是将比较矮的树作为子树，添加到较高的树中。为了保存秩，需要额外使用一个与 uset 同长度的数组，并将所有元素都初始化为 0。
    @Override
    public void union(long id1, long id2) {
        while (true) {
            id1 = find(id1);
            id2 = find(id2);
            if (id1 == id2) {
                return;
            }

            // We need to do Union-by-Min, so the smaller community ID wins.
            // We also only update the entry for id1 and if that
            // is the smaller value, we need to swap ids so we update
            // only the value for id2, not id1.
            if (setIdOf(id1) < setIdOf(id2)) {
                long tmp = id2;
                id2 = id1;
                id1 = tmp;
            }

            long oldEntry = id1;
            long newEntry = id2;

            if (!parent.compareAndSet(id1, oldEntry, newEntry)) {
                continue;
            }

            break;
        }
    }

    @Override
    public long size() {
        return parent.size();
    }
}
