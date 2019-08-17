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
public final class ExceptionAttemptContext<T> implements AttemptContext<T> {
    private final ExecutionException e;
    private final int attemptCount;
    private final long elapsedTimeSinceFirstAttempt;

    public ExceptionAttemptContext(Throwable cause, int attemptCount, long elapsedTimeSinceFirstAttempt) {
        this.e = new ExecutionException(cause);
        this.attemptCount = attemptCount;
        this.elapsedTimeSinceFirstAttempt = elapsedTimeSinceFirstAttempt;
    }

    @Override
    public T get() throws ExecutionException {
        throw e;
    }

    @Override
    public T getResult() {
        throw new IllegalStateException("The attempt resulted in an exception. There is no valid result");
    }

    @Override
    public ExecutionException getException() {
        return e;
    }

    @Override
    public boolean hasResult() {
        return false;
    }

    @Override
    public boolean hasException() {
        return true;
    }

    @Override
    public Throwable getExceptionCause() {
        return e.getCause();
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
