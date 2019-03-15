package com.webank.ai.fate.common.mlmodel.model;

import com.webank.ai.fate.api.serving.ServingProto;

import java.util.HashMap;

import static java.lang.Math.exp;

public class HeteroLRGuest extends HeteroLR {
    private double sigmod(float x) {
        return 1. / (1. + exp(-x));
    }

    @Override
    public HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams) {
        HashMap<String, Object> newInputData = data_transform(inputData);

        HashMap<String, Object> result = new HashMap<>();
        float score = 0;
        for (String key : newInputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += (float) newInputData.get(key) * this.weight.get(key);
            }
        }

        score += this.intercept;

        //score += get_host_predict_result();
        ServingProto.PredictResponse hostPredictResponse = this.getHostPredict((String)predictParams.get("sceneId"), (String)predictParams.get("id"), (String)predictParams.get("modelId"));
        double prob = sigmod(score);

        result.put("prob", prob);

        return result;
//        String strThresholds = "thresholds";
//        if (predictParams.containsKey(strThresholds)) {
//            List<Float> thresholds = (List<Float>) predictParams.get(strThresholds);
//            for (int i = 0; i < thresholds.size(); i++) {
//                if(prob > thresholds[i]){
//
//                }
//            }
//        }
//        return result;
    }
}
