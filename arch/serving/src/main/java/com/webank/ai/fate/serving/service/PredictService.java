package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.serving.PredictionServiceGrpc;
import com.webank.ai.fate.api.serving.ServingProto;
import com.webank.ai.fate.api.serving.ServingProto.PredictRequest;
import com.webank.ai.fate.api.serving.ServingProto.PredictResponse;
import com.webank.ai.fate.common.network.grpc.client.ClientPool;
import com.webank.ai.fate.common.utils.Configuration;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;


public class PredictService extends PredictionServiceGrpc.PredictionServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(PredictRequest req, StreamObserver<PredictResponse> responseObserver){
        float score = 0;
        String myRole;
        PredictResponse.Builder response = PredictResponse.newBuilder();

        response.setModelId(req.getModelId());
        response.setPartId(10001);
        switch (req.getRole()){
            case "guestUser":
                myRole = "guest";
                break;
            case "guest":
                myRole = "host";
                break;
            default:
                myRole = "unknown";
        }
        response.setRole(myRole);

        Map<String, Object> features = new HashMap<>();
        req.getDataMap().forEach((id, f)->{
            f.getFloatDataMap().forEach((k, v)->{
                features.put(k, v);
            });
            f.getStringDataMap().forEach((k, v)->{
                features.put(k, v);
            });
            if (myRole.equals("guest")){
                PredictResponse hostResponse = this.getHostPredict(id, req.getModelId());
                LOGGER.info(hostResponse);
            }
            ServingProto.DataMap.Builder dataBuilder = ServingProto.DataMap.newBuilder();
            dataBuilder.putFloatData("score", 10);
            response.putData(id, dataBuilder.build());
        });

        // get model
        // new thread send fpredict
        // preprocess
        // fpredict
        LOGGER.info(response);
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    private PredictResponse getHostPredict(String userId, String modelId){
        ManagedChannel channel = ClientPool.getChannel(Configuration.getProperty("proxy"));
        PredictRequest.Builder builder = PredictRequest.newBuilder();
        builder.putData(userId, ServingProto.DataMap.newBuilder().putStringData("xxx", "xxx").build());
        builder.setRole("guest");
        builder.setModelId(modelId);
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        PredictResponse response = stub.predict(builder.build());
        return response;
    }
}
