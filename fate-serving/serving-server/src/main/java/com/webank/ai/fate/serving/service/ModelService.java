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

package com.webank.ai.fate.serving.service;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceGrpc;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishRequest;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishResponse;
import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.core.bean.*;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.serving.manger.ModelUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ModelService extends ModelServiceGrpc.ModelServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void publishLoad(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver) {

        Context context =  new BaseContext(new BaseLoggerPrinter());
        context.setActionType(ModelActionType.MODEL_LOAD.name());
        context.preProcess();
        ReturnResult returnResult =null;
        try{
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        returnResult = ModelManager.publishLoadModel(
                new FederatedParty(req.getLocal().getRole(), req.getLocal().getPartyId()),
                ModelUtils.getFederatedRoles(req.getRoleMap()),
                ModelUtils.getFederatedRolesModel(req.getModelMap()));
        builder.setStatusCode(returnResult.getRetcode())
                .setMessage(returnResult.getRetmsg())
                .setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult.getData()).getBytes()));
        builder.setStatusCode(returnResult.getRetcode());
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();}
        finally {
            context.postProcess(req,returnResult);
        }
    }

    @Override
    public void publishOnline(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver) {
        Context context =  new BaseContext(new BaseLoggerPrinter());
        context.setActionType(ModelActionType.MODEL_PUBLISH_ONLINE.name());
        context.preProcess();
        ReturnResult returnResult =null;
        try {
            PublishResponse.Builder builder = PublishResponse.newBuilder();
             returnResult = ModelManager.publishOnlineModel(
                    new FederatedParty(req.getLocal().getRole(), req.getLocal().getPartyId()),
                    ModelUtils.getFederatedRoles(req.getRoleMap()),
                    ModelUtils.getFederatedRolesModel(req.getModelMap())
            );
            builder.setStatusCode(returnResult.getRetcode())
                    .setMessage(returnResult.getRetmsg())
                    .setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult.getData()).getBytes()));
            responseStreamObserver.onNext(builder.build());
            responseStreamObserver.onCompleted();
        }finally {
            context.postProcess(req,returnResult);
        }
    }
}
