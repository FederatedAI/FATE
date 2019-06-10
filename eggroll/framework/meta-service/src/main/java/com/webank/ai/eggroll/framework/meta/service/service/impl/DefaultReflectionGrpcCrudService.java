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

package com.webank.ai.eggroll.framework.meta.service.service.impl;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.core.factory.CallMetaModelFactory;
import com.webank.ai.eggroll.core.helper.ParamValidationHelper;
import com.webank.ai.eggroll.core.serdes.impl.ByteStringSerDesHelper;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.CrudServerProcessor;
import com.webank.ai.eggroll.framework.meta.service.service.GrpcCrudService;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.springframework.beans.factory.annotation.Autowired;

@Deprecated
public class DefaultReflectionGrpcCrudService implements GrpcCrudService {
    @Autowired
    private ByteStringSerDesHelper byteStringSerDesHelper;
    @Autowired
    private DaoServiceFactory daoServiceFactory;
    @Autowired
    private CallMetaModelFactory callMetaModelFactory;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ParamValidationHelper paramValidationHelper;

    private Class recordClass;

    @Override
    public void init(Class recordClass) {
        this.recordClass = recordClass;
    }

    @Override
    public GenericDaoService getGenericDaoService() {
        return null;
    }

    @Override
    public void create(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            Object record = byteStringSerDesHelper.deserialize(request.getParam().getData(), recordClass);
            String methodName = "create" + recordClass.getSimpleName() + "DaoService";
            GenericDaoService genericDaoService = (GenericDaoService) MethodUtils.invokeMethod(daoServiceFactory, methodName);

            int rowsAffected = genericDaoService.insertSelective(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(
                        101, "Failed to create" + recordClass.getSimpleName() + " in database", record);
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(e);
        }
    }

    @Override
    // todo: combine create and update and select code when have time
    public void update(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            Object record = byteStringSerDesHelper.deserialize(request.getParam().getData(), recordClass);
            String methodName = "create" + recordClass.getSimpleName() + "DaoService";
            GenericDaoService genericDaoService = (GenericDaoService) MethodUtils.invokeMethod(daoServiceFactory, methodName);

            int rowsAffected = genericDaoService.updateByPrimaryKey(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(
                        102, "Failed to update" + recordClass.getSimpleName() + " in database", record);
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(e);
        }
    }

    @Override
    public void createOrUpdate(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {

    }

    @Override
    public <T> void processCrudRequest(BasicMeta.CallRequest request, StreamObserver responseObserver, CrudServerProcessor<T> crudServerProcessor) {

    }

    @Override
    public <T> Object processCrudRequest(Object record, CrudServerProcessor<T> crudServerProcessor) throws CrudException {
        return null;
    }

    @Override
    // todo: combine create and update and select code when have time
    public void getById(BasicMeta.CallRequest request, StreamObserver responseObserver) {
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            BasicMeta.Data requestData = request.getParam();
            Object record = byteStringSerDesHelper.deserialize(requestData.getData(), Class.forName(requestData.getType()));
            String methodName = "create" + recordClass.getSimpleName() + "DaoService";
            GenericDaoService genericDaoService = (GenericDaoService) MethodUtils.invokeMethod(daoServiceFactory, methodName);

            Object object = genericDaoService.selectByPrimaryKey(record);

            if (object != null) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(
                        102, "Failed to update" + recordClass.getSimpleName() + " in database", record);
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(e);
        }
    }
}
