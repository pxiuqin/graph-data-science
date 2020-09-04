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
package org.neo4j.graphalgo.beta.generator;

import java.util.Objects;

public interface PropertyProducer {

    static PropertyProducer fixed(String propertyName, double value) {
        return new Fixed(propertyName, value);
    }

    static PropertyProducer random(String propertyName, double min, double max) {
        return new Random(propertyName, min, max);
    }

    String getPropertyName();

    double getPropertyValue(java.util.Random random);

    class Fixed implements PropertyProducer {
        private final String propertyName;
        private final double value;

        public Fixed(String propertyName, double value) {
            this.propertyName = propertyName;
            this.value = value;}

        @Override
        public String getPropertyName() {
            return propertyName;
        }

        @Override
        public double getPropertyValue(java.util.Random random) {
            return value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Fixed fixed = (Fixed) o;
            return Double.compare(fixed.value, value) == 0 &&
                   Objects.equals(propertyName, fixed.propertyName);
        }

        @Override
        public int hashCode() {
            return Objects.hash(propertyName, value);
        }

        @Override
        public String toString() {
            return "Fixed{" +
                   "propertyName='" + propertyName + '\'' +
                   ", value=" + value +
                   '}';
        }
    }

    class Random implements PropertyProducer {
        private final String propertyName;
        private final double min;
        private final double max;

        public Random(String propertyName, double min, double max) {
            this.propertyName = propertyName;
            this.min = min;
            this.max = max;

            if (max <= min) {
                throw new IllegalArgumentException("Max value must be greater than min value");
            }
        }

        @Override
        public String getPropertyName() {
            return propertyName;
        }

        @Override
        public double getPropertyValue(java.util.Random random) {
            return min + (random.nextDouble() * (max - min));
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Random random = (Random) o;
            return Double.compare(random.min, min) == 0 &&
                   Double.compare(random.max, max) == 0 &&
                   Objects.equals(propertyName, random.propertyName);
        }

        @Override
        public int hashCode() {
            return Objects.hash(propertyName, min, max);
        }

        @Override
        public String toString() {
            return "Random{" +
                   "propertyName='" + propertyName + '\'' +
                   ", min=" + min +
                   ", max=" + max +
                   '}';
        }
    }

    class EmptyPropertyProducer implements PropertyProducer {
        @Override
        public String getPropertyName() {
            return null;
        }

        @Override
        public double getPropertyValue(java.util.Random random) {
            return 0;
        }
    }
}
