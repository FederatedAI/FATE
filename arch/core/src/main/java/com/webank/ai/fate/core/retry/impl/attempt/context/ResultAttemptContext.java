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

package com.webank.ai.fate.core.retry.impl.attempt.context;

import com.webank.ai.fate.core.retry.AttemptContext;

import javax.annotation.concurrent.Immutable;
import java.util.concurrent.ExecutionException;

@Immutable
public final class ResultAttemptContext<T> implements AttemptContext<T> {
    private final T result;
    private final int attemptCount;
    private final long elapsedTimeSinceFirstAttempt;

    public ResultAttemptContext(T result, int attemptCount, long elapsedTimeSinceFirstAttempt) {
        this.result = result;
        this.attemptCount = attemptCount;
        this.elapsedTimeSinceFirstAttempt = elapsedTimeSinceFirstAttempt;
    }

    @Override
    public T get() {
        return result;
    }

    @Override
    public T getResult() {
        return result;
    }

    @Override
    public ExecutionException getException() {
        throw new IllegalStateException("The attempt resulted in a result. There is no exception");
    }

    @Override
    public boolean hasResult() {
        return true;
    }

    @Override
    public boolean hasException() {
        return false;
    }

    @Override
    public Throwable getExceptionCause() {
        throw new IllegalStateException("The attempt resulted in a result. There is no exception");
    }

    @Override
    public int getAttemptCount() {
        return attemptCount;
    }

    @Override
    public long getElapsedTimeSinceFirstAttempt() {
        return elapsedTimeSinceFirstAttempt;
    }
}
