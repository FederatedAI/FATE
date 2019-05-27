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

import com.webank.ai.fate.core.retry.StopStrategy;
import com.webank.ai.fate.core.retry.impl.stop.NeverStopStrategy;
import com.webank.ai.fate.core.retry.impl.stop.StopAfterMaxAttemptStrategy;
import com.webank.ai.fate.core.retry.impl.stop.StopAfterMaxDelayStrategy;

public enum StopStrategies {
    INSTANCE;

    private static final StopStrategy NEVER_STOP = new NeverStopStrategy();

    public static StopStrategy neverStop() {
        return NEVER_STOP;
    }

    public static StopStrategy stopAfterMaxAttempt(int maxAttemptNumber) {
        return new StopAfterMaxAttemptStrategy(maxAttemptNumber);
    }

    public static StopStrategy stopAfterMaxDelay(long maxDelayMs) {
        return new StopAfterMaxDelayStrategy(maxDelayMs);
    }
}
