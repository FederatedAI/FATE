/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.core.utils;

import com.google.common.base.Preconditions;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.security.SecureRandom;

@Component
@Scope("prototype")
public class RandomUtils {
    private final SecureRandom RANDOM;

    public RandomUtils() {
        this.RANDOM = new SecureRandom();
    }

    public RandomUtils(byte[] seed) {
        this.RANDOM = new SecureRandom(seed);
    }

    public int nextInt() {
        return nextInt(Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    public int nextNonNegativeInt() {
        return nextInt(0, Integer.MAX_VALUE);
    }

    public int nextInt(int lowerBoundInclusive, int higherBoundInclusive) {
        Preconditions.checkArgument(lowerBoundInclusive <= higherBoundInclusive, "lower bound must be less than or equal to higher bound");
        if (lowerBoundInclusive == higherBoundInclusive) {
            return lowerBoundInclusive;
        }
        return lowerBoundInclusive + RANDOM.nextInt(higherBoundInclusive - lowerBoundInclusive);
    }

    public long nextLong() {
        return nextLong(Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public long nextNonNegativeLong() {
        return nextLong(0L, Long.MAX_VALUE);
    }

    public long nextLong(long lowerBoundInclusive, long higherBoundInclusive) {
        Preconditions.checkArgument(lowerBoundInclusive <= higherBoundInclusive, "lower bound must be less than or equal to higher bound");
        if (lowerBoundInclusive == higherBoundInclusive) {
            return lowerBoundInclusive;
        }
        return (long) nextDouble((double) lowerBoundInclusive, (double) higherBoundInclusive);
    }

    public double nextDouble() {
        return nextDouble(Double.MIN_VALUE, Double.MAX_VALUE);
    }

    public double nextNonNegativeDuoble() {
        return nextDouble(0D, Double.MAX_VALUE);
    }

    public double nextDouble(double lowerBoundInclusive, double higherBoundInclusive) {
        Preconditions.checkArgument(lowerBoundInclusive <= higherBoundInclusive, "lower bound must be less than or equal to higher bound");
        if (lowerBoundInclusive == higherBoundInclusive) {
            return lowerBoundInclusive;
        }
        return lowerBoundInclusive + (higherBoundInclusive - lowerBoundInclusive) * RANDOM.nextDouble();
    }

    public float nextFloat() {
        return nextFloat(Float.MAX_VALUE, Float.MAX_VALUE);
    }

    public float nextNonNegativeFloat() {
        return nextFloat(0F, Float.MAX_VALUE);
    }

    public float nextFloat(float lowerBoundInclusive, float higherBoundInclusive) {
        Preconditions.checkArgument(lowerBoundInclusive <= higherBoundInclusive, "lower bound must be less than or equal to higher bound");
        if (lowerBoundInclusive == higherBoundInclusive) {
            return lowerBoundInclusive;
        }
        return lowerBoundInclusive + (higherBoundInclusive - lowerBoundInclusive) * RANDOM.nextFloat();
    }

}
