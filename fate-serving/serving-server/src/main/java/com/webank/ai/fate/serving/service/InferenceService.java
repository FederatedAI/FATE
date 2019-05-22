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
import com.webank.ai.fate.api.serving.InferenceServiceProto.InferenceMessage;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.bean.InferenceRequest;
import com.webank.ai.fate.serving.manger.InferenceManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


public class InferenceService extends InferenceServiceGrpc.InferenceServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void inference(InferenceMessage req, StreamObserver<InferenceMessage> responseObserver){
        InferenceMessage.Builder response = InferenceMessage.newBuilder();
        InferenceRequest inferenceRequest = (InferenceRequest) ObjectTransform.json2Bean(req.getData().toStringUtf8(), InferenceRequest.class);
        ReturnResult returnResult = InferenceManager.inference(inferenceRequest);
        response.setData(ByteString.copyFrom(ObjectTransform.bean2Json(returnResult).getBytes()));
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }
}
