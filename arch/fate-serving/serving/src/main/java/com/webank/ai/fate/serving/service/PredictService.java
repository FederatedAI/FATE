package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.serving.PredictionServiceGrpc;
import com.webank.ai.fate.api.serving.PredictionServiceProto;
import com.webank.ai.fate.api.serving.PredictionServiceProto.PredictRequest;
import com.webank.ai.fate.api.serving.PredictionServiceProto.PredictResponse;
import com.webank.ai.fate.api.serving.PredictionServiceProto.FederatedMeta;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.statuscode.ReturnCode;
import com.webank.ai.fate.serving.utils.FederatedUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class PredictService extends PredictionServiceGrpc.PredictionServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void predict(PredictRequest req, StreamObserver<PredictResponse> responseObserver){
        FederatedMeta requestMeta = req.getMeta();

        PredictResponse.Builder response = PredictResponse.newBuilder();
        String myRole = FederatedUtils.getMyRole(requestMeta.getRole());

        // get model
        MLModel model = new ModelManager().getModel(requestMeta.getSceneId(), requestMeta.getPartyId(), myRole);
        if (model == null){
            response.setStatusCode(ReturnCode.NOMODEL);
            FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta, "");
            response.setMeta(federatedMetaBuilder.build());
            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
            return;
        }


        FederatedMeta.Builder federatedMetaBuilder = FederatedUtils.genResponseMetaBuilder(requestMeta, (String)model.getModelInfo().get("modelId"));
        // set response meta
        response.setMeta(federatedMetaBuilder.build());
        // deal data
        req.getDataMap().forEach((sid, f)->{
            Map<String, Object> predictInputData = new HashMap<>();
            f.getFloatDataMap().forEach((k, v)->{
                predictInputData.put(k, v);
            });
            f.getStringDataMap().forEach((k, v)->{
                predictInputData.put(k, v);
            });

            Map<String, String> predictParams = new HashMap<>();
            predictParams.put("sceneId", requestMeta.getSceneId());
            predictParams.put("sid", sid);
            predictParams.put("modelId", (String)model.getModelInfo().get("modelId"));
            Map<String, Object> result = model.predict(predictInputData, predictParams);

            PredictionServiceProto.DataMap.Builder dataBuilder = PredictionServiceProto.DataMap.newBuilder();
            dataBuilder.putFloatData("prob", (float)result.get("prob")); // just a test
            response.putData(sid, dataBuilder.build());
        });

        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    public Map<String, Object> federatedPredict(Map<String, Object> requestInfo){
        MLModel model = new ModelManager().getModel((String)requestInfo.get("sceneId"), (String)requestInfo.get("partnerPartyId"), "host");
        Map<String, String> predictParams = new HashMap<>();
        predictParams.put("sceneId", (String)requestInfo.get("sceneId"));
        predictParams.put("sid", (String)requestInfo.get("sid"));
        predictParams.put("modelId", (String)model.getModelInfo().get("modelId"));
        Map<String, Object> result = model.predict(getFeatureData((String)requestInfo.get("sid")), predictParams);
        result.putAll(model.getModelInfo());
        return result;
    }

    private Map<String, Object> getFeatureData(String sid){
        Map<String, Object> featureData = new HashMap<>();
        try{
            List<String> lines = Files.readAllLines(Paths.get(System.getProperty("user.dir"), "host_data.csv"));
            lines.forEach(line->{
                for(String kv: StringUtils.split(line, ",")){
                    String[] a = StringUtils.split(kv, ":");
                    featureData.put(a[0], Float.parseFloat(a[1]));
                }
            });
        }
        catch (Exception ex){
            LOGGER.error(ex);
        }
        return featureData;
    }
}
