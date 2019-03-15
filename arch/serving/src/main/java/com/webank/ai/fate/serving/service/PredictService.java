package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.serving.PredictionServiceGrpc;
import com.webank.ai.fate.api.serving.ServingProto;
import com.webank.ai.fate.api.serving.ServingProto.PredictRequest;
import com.webank.ai.fate.api.serving.ServingProto.PredictResponse;
import com.webank.ai.fate.common.mlmodel.manager.ModelManager;
import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.utils.Configuration;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;


public class PredictService extends PredictionServiceGrpc.PredictionServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(PredictRequest req, StreamObserver<PredictResponse> responseObserver){
        String myRole;
        PredictResponse.Builder response = PredictResponse.newBuilder();

        response.setPartyId(Configuration.getProperty("partyId"));
        response.setSceneId(req.getSceneId());
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

        // get model
        MLModel model = new ModelManager().getModel(req.getSceneId(), req.getPartyId(), myRole);
        response.setModelId(model.getModelId());
        req.getDataMap().forEach((id, f)->{
            Map<String, Object> predictInputData = new HashMap<>();
            f.getFloatDataMap().forEach((k, v)->{
                predictInputData.put(k, v);
            });
            f.getStringDataMap().forEach((k, v)->{
                predictInputData.put(k, v);
            });

            Map<String, String> predictParams = new HashMap<>();
            predictParams.put("sceneId", req.getSceneId());
            predictParams.put("id", id);
            predictParams.put("modelId", model.getModelId());
            Map<String, Object> result = model.predict(predictInputData, predictParams);

            ServingProto.DataMap.Builder dataBuilder = ServingProto.DataMap.newBuilder();
            dataBuilder.putFloatData("prob", (float)result.get("prob")); // just a test
            response.putData(id, dataBuilder.build());
        });

        // new thread send fpredict
        // preprocess
        // fpredict
        LOGGER.info(response);
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }
}
