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
public final class ExponentialWaitTimeStrategy implements WaitTimeStategy {
    private final long multiplier;
    private final long maxWait;

    public ExponentialWaitTimeStrategy(long multiplier, long maxWait) {
        Preconditions.checkArgument(multiplier > 0, "multiplier must > 0");
        Preconditions.checkArgument(maxWait >= 0, "max wait must >= 0");
        Preconditions.checkArgument(multiplier < maxWait, "multiplier must < max wait");

        this.multiplier = multiplier;
        this.maxWait = maxWait;
    }

    @Override
    public long computeCurrentWaitTime(AttemptContext<?> failedAttemptContext) {
        long exp = 1 << (failedAttemptContext.getAttemptCount() - 1);
        long result = multiplier << exp;

        return result <= maxWait ? result : maxWait;
    }
}
