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

@Immutable
public final class ArithmeticProgressionWaitTimeStrategy implements WaitTimeStategy {
    private final long initial;
    private final long commonDifference;        // can be <= 0. but if final result < 0, we return 0

    public ArithmeticProgressionWaitTimeStrategy(long initial, long commonDifference) {
        Preconditions.checkArgument(initial >= 0, "initial wait time must >= 0");

        this.initial = initial;
        this.commonDifference = commonDifference;
    }

    @Override
    public long computeCurrentWaitTime(AttemptContext<?> failedAttemptContext) {
        long result = initial + (commonDifference * (failedAttemptContext.getAttemptCount() - 1));
        return result >= 0L ? result : 0L;
    }
}
