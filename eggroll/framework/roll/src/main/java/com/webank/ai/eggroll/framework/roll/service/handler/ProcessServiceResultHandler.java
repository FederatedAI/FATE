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

package com.webank.ai.eggroll.framework.roll.service.handler;

import io.grpc.stub.StreamObserver;

import java.util.List;

/**
 * process result so that T -> E
 *
 * @param <T> element type in list
 * @param <E> request observer's element type. e.g. calleE type
 *            <p>
 *            It is possible that T and E are from the same type
 */
public interface ProcessServiceResultHandler<T, E> {
    public void handle(StreamObserver<E> requestObserver, List<T> unprocessedResults) throws Throwable;
}
