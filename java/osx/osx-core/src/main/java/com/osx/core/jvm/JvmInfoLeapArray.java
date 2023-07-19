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

package com.osx.core.jvm;

import com.osx.core.flow.LeapArray;
import com.osx.core.flow.WindowWrap;

public class JvmInfoLeapArray extends LeapArray<JvmInfo> {

    public JvmInfoLeapArray(int sampleCount, int intervalInMs) {
        super(sampleCount, intervalInMs);
    }

    @Override
    public JvmInfo newEmptyBucket(long timeMillis) {
        return new JvmInfo(timeMillis);
    }

    @Override
    protected WindowWrap<JvmInfo> resetWindowTo(WindowWrap<JvmInfo> windowWrap, long startTime) {
        windowWrap.resetTo(startTime);
        windowWrap.value().reset();
        return windowWrap;
    }
}
