package com.webank.ai.fate.serving.federatedml.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.exp;

public class HeteroLRGuest extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();
    private double sigmod(float x) {
        return 1. / (1. + exp(-x));
    }

    @Override
    public HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams){
        HashMap<String, Object> newInputData = data_transform(inputData);

        HashMap<String, Object> result = new HashMap<>();
        float score = forward(newInputData);
        LOGGER.info("guest score:{}", score);

        try{
            Map<String, Object> hostPredictResponse = this.getHostPredict((String)predictParams.get("sid"));
            score += (double)hostPredictResponse.get("score");
        }
        catch (Exception ex){
            LOGGER.error(ex);
        }

        double prob = sigmod(score);
        result.put("prob", (float)prob);

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
