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
import com.webank.ai.fate.api.serving.InferenceServiceGrpc;
import com.webank.ai.fate.api.serving.InferenceServiceProto.InferenceRequest;
import com.webank.ai.fate.api.serving.InferenceServiceProto.InferenceResponse;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.manger.ModelInfo;
import com.webank.ai.fate.serving.manger.ModelUtils;
import com.webank.ai.fate.serving.manger.InferenceManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.util.List;
import java.util.Map;


public class InferenceService extends InferenceServiceGrpc.InferenceServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(InferenceRequest req, StreamObserver<InferenceResponse> responseObserver){
        InferenceResponse.Builder response = InferenceResponse.newBuilder();
        Map<String, List<Integer>> allParty = ModelUtils.getAllParty(req.getRoleMap());
        ReturnResult returnResult = InferenceManager.inference(req.getLocal().getRole(),
                req.getLocal().getPartyId(),
                allParty,
                new ModelInfo(req.getModel().getTableName(), req.getModel().getNamespace()),
                req.getData().toStringUtf8(),
                req.getSceneId());
        response.setStatusCode(returnResult.getStatusCode());
        response.setMessage(returnResult.getMessage());
        if (returnResult.getData() != null){
            response.setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult.getData()).getBytes()));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }
}
