package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.serving.PredictionServiceGrpc;
import com.webank.ai.fate.api.serving.ServingProto;
import com.webank.ai.fate.api.serving.ServingProto.PredictRequest;
import com.webank.ai.fate.api.serving.ServingProto.PredictResponse;
import io.grpc.stub.StreamObserver;

import java.util.HashMap;
import java.util.Map;


public class PredictService extends PredictionServiceGrpc.PredictionServiceImplBase {

    @Override
    public void predict(PredictRequest req, StreamObserver<PredictResponse> responseObserver){
        float score = 0;
        PredictResponse.Builder response = PredictResponse.newBuilder();

        response.setModelId(req.getModelId());
        response.setPartId(10001);
        response.setRole("host");

        Map<String, Object> features = new HashMap<>();
        req.getDataMap().forEach((id, f)->{
            f.getFloatDataMap().forEach((k, v)->{
                features.put(k, v);
            });
            f.getStringDataMap().forEach((k, v)->{
                features.put(k, v);
            });
            ServingProto.DataMap.Builder dataBuilder = ServingProto.DataMap.newBuilder();
            dataBuilder.putFloatData("score", 10);
            response.putData(id, dataBuilder.build());
        });

        // get model
        // new thread send fpredict
        // preprocess
        // fpredict
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }
}
