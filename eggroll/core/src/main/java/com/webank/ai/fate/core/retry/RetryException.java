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

package com.webank.ai.fate.core.retry;

public class RetryException extends Exception {
    private final int numberOfFailedAttempts;
    private final AttemptContext<?> lastFailedAttemptContext;

    public RetryException(int numberOfFailedAttempts, AttemptContext<?> lastFailedAttemptContext) {
        this("Retry failed. Number from attempts: " + numberOfFailedAttempts, numberOfFailedAttempts,
                lastFailedAttemptContext);
    }

    public RetryException(String message, int numberOfFailedAttempts, AttemptContext<?> lastFailedAttemptContext) {
        super(message, lastFailedAttemptContext == null ? null : lastFailedAttemptContext.getExceptionCause());
        this.numberOfFailedAttempts = numberOfFailedAttempts;
        this.lastFailedAttemptContext = lastFailedAttemptContext;
    }

    public int getNumberOfFailedAttempts() {
        return numberOfFailedAttempts;
    }

    public AttemptContext<?> getLastFailedAttemptContext() {
        return lastFailedAttemptContext;
    }
}
