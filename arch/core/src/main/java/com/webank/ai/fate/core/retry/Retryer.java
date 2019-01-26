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

import com.google.common.base.Predicate;
import com.webank.ai.fate.core.retry.impl.attempt.context.ExceptionAttemptContext;
import com.webank.ai.fate.core.retry.impl.attempt.context.ResultAttemptContext;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;

public class Retryer<T> {
    private static final Logger LOGGER = LogManager.getLogger(Retryer.class);
    private final AttemptOperation<T> attemptOperation;
    private final StopStrategy stopStrategy;
    private final WaitStrategy waitStrategy;
    private final WaitTimeStategy waitTimeStategy;
    private final Collection<Predicate<AttemptContext<T>>> resultRejectionPredicates;
    private final Collection<RetryListener> retryListeners;
    private String description;

    public Retryer(AttemptOperation<T> attemptOperation,
                   StopStrategy stopStrategy,
                   WaitStrategy waitStrategy,
                   WaitTimeStategy waitTimeStategy,
                   Collection<Predicate<AttemptContext<T>>> resultRejectionPredicates,
                   Collection<RetryListener> retryListeners) {
        this.attemptOperation = attemptOperation;
        this.stopStrategy = stopStrategy;
        this.waitStrategy = waitStrategy;
        this.waitTimeStategy = waitTimeStategy;
        this.resultRejectionPredicates = resultRejectionPredicates;
        this.retryListeners = retryListeners;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public T call(Callable<T> callable) throws ExecutionException, RetryException {
        long startTime = System.currentTimeMillis();

        // 1. here we go
        for (int attemptCount = 1; ; ++attemptCount) {
            AttemptContext<T> attemptContext;

            if (!StringUtils.isBlank(description)) {
                LOGGER.info("attempt count: {}; {}", attemptCount, description);
            }

            // 2. actual call
            try {
                T result = attemptOperation.call(callable);

                // 2.1. successfully called and returned: init context with result
                attemptContext = new ResultAttemptContext<T>(
                        result,
                        attemptCount,
                        System.currentTimeMillis() - startTime);
            } catch (Throwable t) {
                // 2.2. throwable caught in actual call: int context with exception
                attemptContext = new ExceptionAttemptContext<T>(
                        t,
                        attemptCount,
                        System.currentTimeMillis() - startTime);
            }

            // 3. call hooks / listeners registered
            for (RetryListener retryListener : retryListeners) {
                retryListener.onRetry(attemptContext);
            }

            // 4. if it is a result, check if it showld be rejected. if no one rejects, then return it
            boolean isResultOk = true;
            for (Predicate<AttemptContext<T>> resultRejectionPredicate : resultRejectionPredicates) {
                if (resultRejectionPredicate.apply(attemptContext)) {
                    isResultOk = false;
                    break;
                }
            }

            if (isResultOk) {
                return attemptContext.get();
            }

            // 5. check if retry should stop
            if (stopStrategy.shouldStop(attemptContext)) {
                throw new RetryException(attemptCount, attemptContext);
            } else {
                // 6. retry continues. we need to compute current wait time
                long waitTime = waitTimeStategy.computeCurrentWaitTime(attemptContext);

                try {
                    // 7. wait starts
                    waitStrategy.startWait(waitTime);
                } catch (InterruptedException e) {
                    // interrupted, retry exceptions are thrown
                    Thread.currentThread().interrupt();
                    throw new RetryException(attemptCount, attemptContext);
                }
            }
        }


    }
}
