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
package org.neo4j.graphalgo.core.utils.queue;

import org.neo4j.graphalgo.core.utils.mem.MemoryEstimation;
import org.neo4j.graphalgo.core.utils.mem.MemoryEstimations;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.LongStream;

import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfDoubleArray;
import static org.neo4j.graphalgo.core.utils.mem.MemoryUsage.sizeOfLongArray;

//有界优先队列
public abstract class BoundedLongLongPriorityQueue {

    public interface Consumer {
        void accept(long element1, long element2, double priority);
    }

    public static MemoryEstimation memoryEstimation(int capacity) {
        return MemoryEstimations.builder(BoundedLongLongPriorityQueue.class)
            .fixed("elements1", sizeOfLongArray(capacity))
            .fixed("elements2", sizeOfLongArray(capacity))
            .fixed("priorities", sizeOfDoubleArray(capacity))
            .build();
    }

    private final int bound;  //队列的界
    private double minValue = Double.NaN;  //优先级最小为NaN

    final long[] elements1;
    final long[] elements2;
    final double[] priorities;
    int elementCount = 0;  //队列元素大小

    BoundedLongLongPriorityQueue(int bound) {
        this.bound = bound;
        this.elements1 = new long[bound];
        this.elements2 = new long[bound];
        this.priorities = new double[bound];
    }

    public abstract boolean offer(long element1, long element2, double priority);

    public abstract void foreach(Consumer consumer);

    public int size() {
        return elementCount;
    }

    //基于一组元素到队列中
    protected boolean add(long element1, long element2, double priority) {
        if (elementCount < bound || Double.isNaN(minValue) || priority < minValue) {
            //因为一开始就使用了二分查找，所以数据是排序的
            int idx = Arrays.binarySearch(priorities, 0, elementCount, priority);  //找对所在优先级下标
            idx = (idx < 0) ? -idx : idx + 1;  //如果没有找到优先级标识，那么设置为队头
            int length = bound - idx;  //计算需要移动的长度
            if (length > 0 && idx < bound) {
                System.arraycopy(priorities, idx - 1, priorities, idx, length);  //腾出idx-1的位置
                System.arraycopy(elements1, idx - 1, elements1, idx, length);
                System.arraycopy(elements2, idx - 1, elements2, idx, length);
            }
            priorities[idx - 1] = priority;  //在腾出的位置赋值
            elements1[idx - 1] = element1;
            elements2[idx - 1] = element2;
            if (elementCount < bound) {
                elementCount++;  //找到最后位置
            }
            minValue = priorities[elementCount - 1];  //找到最小优先级
            return true;
        }
        return false;
    }

    public LongStream elements1() {
        return elementCount == 0
            ? LongStream.empty()
            : Arrays.stream(elements1).limit(elementCount);
    }

    public LongStream elements2() {
        return elementCount == 0
            ? LongStream.empty()
            : Arrays.stream(elements2).limit(elementCount);
    }

    public DoubleStream priorities() {
        return elementCount == 0
            ? DoubleStream.empty()
            : Arrays.stream(priorities).limit(elementCount);
    }

    //使用优先级加负数来实现反向取最大
    public static BoundedLongLongPriorityQueue max(int bound) {
        return new BoundedLongLongPriorityQueue(bound) {

            @Override
            public boolean offer(long element1, long element2, double priority) {
                return add(element1, element2, -priority);  //放到最前面[越小越是最前，出队的时候就是最大了]，offer主动提前
            }

            @Override
            public void foreach(Consumer consumer) {
                for (int i = 0; i < elementCount; i++) {
                    consumer.accept(elements1[i], elements2[i], -priorities[i]); //为啥是负数，和offer形成呼应
                }
            }

            @Override
            public DoubleStream priorities() {
                return elementCount == 0
                    ? DoubleStream.empty()
                    : Arrays.stream(priorities).map(d -> -d).limit(elementCount);  //也是转换成负值，因为入队的时候统一为负数了
            }
        };
    }

    //正常情况下就是min的方式
    public static BoundedLongLongPriorityQueue min(int bound) {
        return new BoundedLongLongPriorityQueue(bound) {

            @Override
            public boolean offer(long element1, long element2, double priority) {
                return add(element1, element2, priority);  //不取负数就是按照实际大小来在队列排序了，这样取出的就是最小，所以是min
            }

            @Override
            public void foreach(Consumer consumer) {
                for (int i = 0; i < elementCount; i++) {
                    consumer.accept(elements1[i], elements2[i], priorities[i]);
                }
            }
        };
    }
}
