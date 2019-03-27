package com.webank.ai.fate.serving.federatedml.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import com.webank.ai.fate.serving.federatedml.transform.DataTransform;

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
        DataTransform dataTransform = new DataTransform();
        HashMap<String, Object> newInputData = dataTransform.fit(inputData, this.dataTransformServer);

        HashMap<String, Object> result = new HashMap<>();
        float score = forward(newInputData);
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
