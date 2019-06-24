package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.bean.ReturnResult;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.exp;

public class HeteroLRGuest extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();

    private double sigmod(double x) {
        return 1. / (1. + exp(-x));
    }

    @Override
    public Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams) {
        Map<String, Object> result = new HashMap<>();
        Map<String, Double> forwardRet = forward(inputData);
        double score = forwardRet.get("score");
        LOGGER.info("guest score:{}", score);

        try {
            ReturnResult hostPredictResponse = this.getFederatedPredict((Map<String, Object>) predictParams.get("federatedParams"));
            predictParams.put("federatedResult", hostPredictResponse);
            double hostScore = (double) hostPredictResponse.getData().get("score");
            LOGGER.info("host score:{}", hostScore);
            score += hostScore;
        } catch (Exception ex) {
            LOGGER.error("get host predict failed:", ex);
        }

        double prob = sigmod(score);
        result.put("prob", prob);
        result.put("guestModelWeightHitRate:{}", forwardRet.get("modelWrightHitRate"));
        result.put("guestInputDataHitRate:{}", forwardRet.get("inputDataHitRate"));

        return result;
    }
}
