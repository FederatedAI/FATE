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

package com.webank.ai.fate.core.retry.impl.waittime;

import com.google.common.base.Preconditions;
import com.webank.ai.fate.core.retry.AttemptContext;
import com.webank.ai.fate.core.retry.WaitTimeStategy;

import javax.annotation.concurrent.Immutable;
import java.security.SecureRandom;

@Immutable
public final class RandomWaitTimeStrategy implements WaitTimeStategy {
    private static final SecureRandom RANDOM = new SecureRandom();
    private final long minMs;
    private final long delta;

    public RandomWaitTimeStrategy(long minMs, long maxMs) {
        Preconditions.checkArgument(minMs >= 0, "wait time must >= 0");
        Preconditions.checkArgument(maxMs > minMs, "max wait time must > min wait time");

        this.minMs = minMs;
        this.delta = (maxMs - minMs);
    }

    @Override
    public long computeCurrentWaitTime(AttemptContext<?> failedAttemptContext) {
        long nl = RANDOM.nextLong();
        long t = Math.abs(nl);
        if (t == Long.MIN_VALUE) {
            t = 0;
        }
        t = t % delta;

        return t + minMs;
    }
}
