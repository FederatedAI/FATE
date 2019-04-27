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
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishRequest;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishResponse;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.manger.ModelInfo;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.serving.manger.ModelUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelService extends ModelServiceGrpc.ModelServiceImplBase{
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void publishLoad(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver){
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        int loadStatus;
        if (Configuration.getPropertyInt("party.id").equals(req.getLocal().getPartyId())){
            ReturnResult returnResult = ModelManager.publishLoadModel(
                    req.getLocal().getRole(),
                    req.getLocal().getPartyId(),
                    ModelUtils.getAllParty(req.getRoleMap()),
                    ModelUtils.getAllPartyModel(req.getModelMap()));
            loadStatus = returnResult.getStatusCode();
            builder.setStatusCode(returnResult.getStatusCode())
                    .setMessage(returnResult.getMessage())
                    .setError(returnResult.getError())
                    .setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult.getData()).getBytes()));
        }
        else{
            loadStatus = StatusCode.NOTME;
        }
        builder.setStatusCode(loadStatus);
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }

    @Override
    public void publishOnline(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver) {
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        ReturnResult returnResult = ModelManager.publishOnlineModel(
                req.getLocal().getRole(),
                req.getLocal().getPartyId(),
                ModelUtils.getAllParty(req.getRoleMap()),
                ModelUtils.getAllPartyModel(req.getModelMap()));
        builder.setStatusCode(returnResult.getStatusCode())
                .setMessage(returnResult.getMessage())
                .setError(returnResult.getError())
                .setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult.getData()).getBytes()));
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }
}
