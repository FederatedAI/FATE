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
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.webank.ai.fate.core.retry.*;
import com.webank.ai.fate.core.retry.impl.predicate.ExceptionClassPredicate;
import com.webank.ai.fate.core.retry.impl.predicate.ExceptionPredicate;
import com.webank.ai.fate.core.retry.impl.predicate.ResultPredicate;
import com.webank.ai.fate.core.retry.*;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

public class RetryerBuilder<T> {
    private AttemptOperation<T> attemptOperation;
    private String description;
    private StopStrategy stopStrategy;
    private WaitStrategy waitStrategy;
    private WaitTimeStategy waitTimeStategy;
    private List<Predicate<AttemptContext<T>>> resultRejectionPredicates;
    private List<RetryListener> retryListeners;

    private Predicate<AttemptContext<T>> defaultRejectionPredicate = Predicates.alwaysFalse();

    private RetryerBuilder() {
        resultRejectionPredicates = new ArrayList<Predicate<AttemptContext<T>>>();
        retryListeners = new ArrayList<RetryListener>();

        defaultRejectionPredicate = Predicates.alwaysFalse();
    }

    public static <T> RetryerBuilder<T> newBuilder() {
        return new RetryerBuilder<T>();
    }

    public RetryerBuilder<T> withAttemptOperation(AttemptOperation<T> attemptOperation) {
        Preconditions.checkNotNull(attemptOperation, "attemptOperation cannot be null");
        this.attemptOperation = attemptOperation;
        return this;
    }

    public RetryerBuilder<T> withDescription(String description) {
        Preconditions.checkState(StringUtils.isBlank(this.description),
                "a description has already been set: " + this.description);
        this.description = description;
        return this;
    }

    public RetryerBuilder<T> withStopStrategy(StopStrategy stopStrategy) {
        Preconditions.checkNotNull(stopStrategy, "stop strategy cannot be null");
        Preconditions.checkState(this.stopStrategy == null,
                "a stop strategy has already been set: " + this.stopStrategy);

        this.stopStrategy = stopStrategy;
        return this;
    }

    public RetryerBuilder<T> withWaitStrategy(WaitStrategy waitStrategy) {
        Preconditions.checkNotNull(waitStrategy, "wait strategy cannot be null");
        Preconditions.checkState(this.waitStrategy == null,
                "a wait strategy has already been set: " + this.waitStrategy);

        this.waitStrategy = waitStrategy;
        return this;
    }

    public RetryerBuilder<T> withWaitTimeStrategy(WaitTimeStategy waitTimeStrategy) {
        Preconditions.checkNotNull(waitTimeStrategy, "wait time strategy cannot be null");
        Preconditions.checkState(this.waitTimeStategy == null,
                "a wait strategy has already been set: " + this.waitTimeStategy);

        this.waitTimeStategy = waitTimeStrategy;
        return this;
    }

    public RetryerBuilder<T> withRetryListener(RetryListener retryListener) {
        Preconditions.checkNotNull(retryListener, "retry listener cannot be null");
        retryListeners.add(retryListener);

        return this;
    }

    public RetryerBuilder<T> withPredicate(Predicate<AttemptContext<T>> predicate) {
        Preconditions.checkNotNull(predicate, "predicate cannot be null");
        resultRejectionPredicates.add(predicate);

        return this;
    }

    public RetryerBuilder<T> retryIfAnyException() {
        retryIfExceptionOfType(Exception.class);
        return this;
    }

    public RetryerBuilder<T> retryIfRuntimeException() {
        retryIfExceptionOfType(RuntimeException.class);
        return this;
    }

    public RetryerBuilder<T> retryIfExceptionOfType(Class<? extends Throwable> exceptionClass) {
        Preconditions.checkNotNull(exceptionClass, "exception class cannot be null");
        defaultRejectionPredicate = Predicates.or(defaultRejectionPredicate,
                new ExceptionClassPredicate<T>(exceptionClass));

        return this;
    }

    public RetryerBuilder<T> retryIfResult(Predicate<T> resultPredicate) {
        Preconditions.checkNotNull(resultPredicate, "result predicate cannot be null");
        defaultRejectionPredicate = Predicates.or(
                defaultRejectionPredicate, new ResultPredicate<T>(resultPredicate));

        return this;
    }

    public RetryerBuilder<T> retryIfException(Predicate<Throwable> exceptionPredicate) {
        Preconditions.checkNotNull(exceptionPredicate, "exception predicate cannot be null");
        defaultRejectionPredicate = Predicates.or(
                defaultRejectionPredicate, new ExceptionPredicate<T>(exceptionPredicate));

        return this;
    }

    public RetryerBuilder<T> retryIfPredicateIsTrue(Predicate<AttemptContext<T>> predicate) {
        Preconditions.checkNotNull(predicate, "exception predicate cannot be null");
        defaultRejectionPredicate = Predicates.or(predicate);

        return this;
    }

    public RetryerBuilder<T> retryIfExpressionIsTrue(boolean expression) {
        if (expression == true) {
            defaultRejectionPredicate = Predicates.or(Predicates.<AttemptContext<T>>alwaysTrue());
        }

        return this;
    }

    public Retryer<T> build() {
        AttemptOperation<T> builtAttemptOperation =
                attemptOperation == null ? AttemptOperations.<T>noTimeLimit() : attemptOperation;
        StopStrategy builtStopStrategy =
                stopStrategy == null ? StopStrategies.neverStop() : stopStrategy;
        WaitStrategy builtWaitStrategy =
                waitStrategy == null ? WaitStrategies.threadSleepWait() : waitStrategy;
        WaitTimeStategy builtWaitTimeStategy =
                waitTimeStategy == null ? WaitTimeStrategies.neverWait() : waitTimeStategy;

        resultRejectionPredicates.add(defaultRejectionPredicate);

        return new Retryer<T>(builtAttemptOperation,
                builtStopStrategy,
                builtWaitStrategy,
                builtWaitTimeStategy,
                resultRejectionPredicates,
                retryListeners);
    }
}
