package com.webank.ai.fate.common.mlmodel.model;


import java.util.Map;

import com.webank.ai.fate.common.network.grpc.client.ClientPool;
import com.webank.ai.fate.api.serving.PredictionServiceGrpc;
import com.webank.ai.fate.api.serving.ServingProto;
import com.webank.ai.fate.api.serving.ServingProto.PredictRequest;
import com.webank.ai.fate.api.serving.ServingProto.PredictResponse;
import com.webank.ai.fate.common.statuscode.ReturnCode;
import com.webank.ai.fate.common.utils.Configuration;
import io.grpc.ManagedChannel;

public abstract class BaseModel<B, X, P> implements MLModel<B, X, P>{
    private String modelId;

    @Override
    public int setModelId(String modelId) {
        this.modelId = modelId;
        return ReturnCode.OK;
    }

    @Override
    public String getModelId() {
        return this.modelId;
    }

    public abstract int initModel(B modelBuffer);
    public abstract Map<String, Object> predict(X inputData, P predictParams);

    protected PredictResponse getHostPredict(String sceneId, String id, String modelId){
        PredictRequest.Builder builder = PredictRequest.newBuilder();
        builder.setPartyId(Configuration.getProperty("partyId"));
        builder.setSceneId(sceneId);
        builder.setRole("guest");
        builder.setModelId(modelId);
        builder.putData(id, ServingProto.DataMap.newBuilder().putStringData("id", "12345678").build());

        ManagedChannel channel = ClientPool.getChannel(Configuration.getProperty("proxy"));
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        PredictResponse response = stub.predict(builder.build());
        return response;
    }
}
