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

package com.webank.ai.eggroll.framework.meta.service.api.grpc.server;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.meta.service.TaskMetaServiceGrpc;
import com.webank.ai.eggroll.core.factory.CallMetaModelFactory;
import com.webank.ai.eggroll.core.helper.ParamValidationHelper;
import com.webank.ai.eggroll.core.serdes.impl.ByteStringSerDesHelper;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Result;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.ResultExample;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Task;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.TaskExample;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class TaskMetaServiceImpl extends TaskMetaServiceGrpc.TaskMetaServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger(TaskMetaServiceImpl.class);
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

    @Override
    public void createTask(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {
        LOGGER.info("Creating task: {}", toStringUtils.toOneLineString(request));
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            Task record = byteStringSerDesHelper.deserialize(request.getParam().getData(), Task.class);

            GenericDaoService<Task, TaskExample, Long> genericDaoService = daoServiceFactory.createTaskDaoService();
            int rowsAffected = genericDaoService.insertSelective(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(101, "Failed to create object in database", record);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(e);
        }
    }

    @Override
    public void updateTask(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {
        LOGGER.info("Updating task: {}", toStringUtils.toOneLineString(request));
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            Task record = byteStringSerDesHelper.deserialize(request.getParam().getData(), Task.class);

            GenericDaoService<Task, TaskExample, Long> genericDaoService = daoServiceFactory.createTaskDaoService();
            int rowsAffected = genericDaoService.updateByPrimaryKey(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(101, "Failed to create object in database", record);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(e);
        }
    }

    @Override
    public void createResult(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {
        LOGGER.info("Creating result: {}", toStringUtils.toOneLineString(request));
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            GenericDaoService<Result, ResultExample, Long> genericDaoService = daoServiceFactory.createResultDaoService();
            Result record = byteStringSerDesHelper.deserialize(request.getParam().getData(), Result.class);
            int rowsAffected = genericDaoService.insertSelective(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(101, "Failed to create result in database", null);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(e);
        }
    }

    @Override
    public void updateResult(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {
        LOGGER.info("Updating result: {}", toStringUtils.toOneLineString(request));
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            GenericDaoService<Result, ResultExample, Long> genericDaoService = daoServiceFactory.createResultDaoService();
            Result record = byteStringSerDesHelper.deserialize(request.getParam().getData(), Result.class);
            int rowsAffected = genericDaoService.updateByPrimaryKey(record);

            if (rowsAffected > 0) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(101, "Failed to update result in database", null);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(e);
        }
    }

    @Override
    public void getResultById(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {
        LOGGER.info("Getting result: {}", toStringUtils.toOneLineString(request));
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            GenericDaoService<Result, ResultExample, Long> genericDaoService = daoServiceFactory.createResultDaoService();
            Long resultId = byteStringSerDesHelper.deserialize(request.getParam().getData(), Long.class);
            Result record = genericDaoService.selectByPrimaryKey(resultId);

            if (record != null) {
                result = callMetaModelFactory.createNormalCallResponse(record);
            } else {
                result = callMetaModelFactory.createErrorCallResponse(101, "Failed to create object in database", resultId);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(e);
        }
    }

}
