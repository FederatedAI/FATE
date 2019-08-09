package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.serving.core.bean.Context;
import com.webank.ai.fate.serving.core.bean.Dict;
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
    public Map<String, Object> predict(Context context , Map<String, Object> inputData, Map<String, Object> predictParams) {
        Map<String, Object> result = new HashMap<>();
        Map<String, Double> forwardRet = forward(inputData);
        double score = forwardRet.get(Dict.SCORE);
        LOGGER.info("caseid {} guest score:{}", context.getCaseId(),score);

        try {
            ReturnResult hostPredictResponse = this.getFederatedPredict(context,(Map<String, Object>) predictParams.get("federatedParams"));
            predictParams.put(Dict.FEDERATED_RESULT, hostPredictResponse);
            context.setFederatedResult(hostPredictResponse);
            double hostScore =0;
            if(hostPredictResponse!=null&&hostPredictResponse.getData()!=null&&hostPredictResponse.getData().get("score")!=null) {
                 hostScore = (double) hostPredictResponse.getData().get(Dict.SCORE);
                LOGGER.info("caseid {} host score:{}", context.getCaseId(), hostScore);
            }else{
                LOGGER.info("caseid {} has no host score",  hostScore);
            }
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
