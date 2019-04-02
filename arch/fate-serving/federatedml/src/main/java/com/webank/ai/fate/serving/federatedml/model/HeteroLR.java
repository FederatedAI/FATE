package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.LRModelParamProto.LRModelParam;

import java.util.Map;

public abstract class HeteroLR extends BaseModel {
    private Map<String, Double> weight;
    private Double intercept;

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        try {
            LRModelParam lrModelParam = LRModelParam.parseFrom(protoParam);

            this.weight = lrModelParam.getWeightMap();
            this.intercept = lrModelParam.getIntercept();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        return StatusCode.OK;
    }

    float forward(Map<String, Object> inputData) {
        float score = 0;
        for (String key : inputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += (float) inputData.get(key) * this.weight.get(key);
            }
        }
        score += this.intercept;

        return score;
    }

    @Override
    public abstract Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams);
}
