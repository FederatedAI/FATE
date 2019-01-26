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

package com.webank.ai.fate.core.retry.factory;

import com.google.common.base.Preconditions;
import com.webank.ai.fate.core.retry.WaitTimeStategy;
import com.webank.ai.fate.core.retry.impl.waittime.ArithmeticProgressionWaitTimeStrategy;
import com.webank.ai.fate.core.retry.impl.waittime.ExponentialWaitTimeStrategy;
import com.webank.ai.fate.core.retry.impl.waittime.FixedWaitTimeStrategy;
import com.webank.ai.fate.core.retry.impl.waittime.RandomWaitTimeStrategy;

import java.util.concurrent.TimeUnit;

public enum WaitTimeStrategies {
    INSTANCE;

    private static final WaitTimeStategy NEVER_WAIT = new FixedWaitTimeStrategy(0L);

    public static WaitTimeStategy neverWait() {
        return NEVER_WAIT;
    }

    public static WaitTimeStategy fixedWaitTime(long waitTimeMs) {
        return fixedWaitTime(waitTimeMs, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy fixedWaitTime(long waitTime, TimeUnit timeUnit) {
        Preconditions.checkNotNull(timeUnit, "time unit cannot be null");
        return new FixedWaitTimeStrategy(timeUnit.toMillis(waitTime));
    }

    public static WaitTimeStategy randomWaitTime(long maxWaitTimeMs) {
        return randomWaitTime(0, TimeUnit.MILLISECONDS, maxWaitTimeMs, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy randomWaitTime(long maxWaitTime, TimeUnit timeUnit) {
        return randomWaitTime(0, TimeUnit.MILLISECONDS, maxWaitTime, timeUnit);
    }

    public static WaitTimeStategy randomWaitTime(long minWaitTimeMs, long maxWaitTimeMs) {
        return randomWaitTime(minWaitTimeMs, TimeUnit.MILLISECONDS, maxWaitTimeMs, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy randomWaitTime(long minWaitTime,
                                                 TimeUnit minWaitTimeUnit,
                                                 long maxWaitTime,
                                                 TimeUnit maxWaitTimeUnit) {
        Preconditions.checkNotNull(minWaitTimeUnit, "min wait time unit cannot be null");
        Preconditions.checkNotNull(maxWaitTimeUnit, "max wait time unit cannot be null");

        return new RandomWaitTimeStrategy(minWaitTimeUnit.toMillis(minWaitTime), maxWaitTimeUnit.toMillis(maxWaitTime));
    }

    public static WaitTimeStategy fixedIncrementWaitTime(long initialMs, long incrementMs) {
        return fixedIncrementWaitTime(initialMs, TimeUnit.MILLISECONDS, incrementMs, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy fixedIncrementWaitTime(long initial,
                                                         TimeUnit initialTimeUnit,
                                                         long increment,
                                                         TimeUnit incrementTimeUnit) {
        Preconditions.checkNotNull(initialTimeUnit, "initial time unit cannot be null");
        Preconditions.checkNotNull(incrementTimeUnit, "increment time unit cannot be null");
        return new ArithmeticProgressionWaitTimeStrategy(
                initialTimeUnit.toMillis(initial),
                incrementTimeUnit.toMillis(increment));
    }


    public static WaitTimeStategy exponentialWaitTime() {
        return exponentialWaitTime(1, Long.MAX_VALUE, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy exponentialWaitTime(long maxWaitTimeMs) {
        return exponentialWaitTime(1, maxWaitTimeMs, TimeUnit.MILLISECONDS);
    }

    public static WaitTimeStategy exponentialWaitTime(long multiplier, long maxWaitTime, TimeUnit maxTimeUnit) {
        Preconditions.checkNotNull(maxTimeUnit, "max time unit cannot be null");
        return new ExponentialWaitTimeStrategy(multiplier, maxTimeUnit.toMillis(maxWaitTime));
    }
}
