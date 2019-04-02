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
    public Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams){
        HashMap<String, Object> result = new HashMap<>();
        float score = forward(inputData);
        LOGGER.info("guest score:{}", score);

        try{
            Map<String, Object> hostPredictResponse = this.getFederatedPredict(predictParams);
            score += (double)hostPredictResponse.get("score");
        }
        catch (Exception ex){
            LOGGER.error(ex);
        }

        double prob = sigmod(score);
        result.put("prob", (float)prob);

        return result;
    }
}
