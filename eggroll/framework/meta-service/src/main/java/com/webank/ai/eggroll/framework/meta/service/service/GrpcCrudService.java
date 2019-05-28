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

package com.webank.ai.eggroll.framework.meta.service.service;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import io.grpc.stub.StreamObserver;

public interface GrpcCrudService {
    public void init(Class recordClass);

    GenericDaoService getGenericDaoService();

    public void create(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver);

    public void update(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver);

    public void createOrUpdate(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver);

    public void getById(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver);

    public <T> void processCrudRequest(BasicMeta.CallRequest request,
                                       StreamObserver responseObserver,
                                       CrudServerProcessor<T> crudServerProcessor);

    public <T> Object processCrudRequest(Object record, CrudServerProcessor<T> crudServerProcessor) throws CrudException;
}
