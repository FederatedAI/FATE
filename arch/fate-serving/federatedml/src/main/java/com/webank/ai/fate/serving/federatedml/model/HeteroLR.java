package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.LRModelParamProto.LRModelParam;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Map;

public abstract class HeteroLR extends BaseModel {
    private Map<String, Double> weight;
    private Double intercept;
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        LOGGER.info("start init HeteroLR class");
        try {
            LRModelParam lrModelParam = LRModelParam.parseFrom(protoParam);

            this.weight = lrModelParam.getWeightMap();
            this.intercept = lrModelParam.getIntercept();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        LOGGER.info("Finish init HeteroLR class");
        return StatusCode.OK;
    }

    double forward(Map<String, Object> inputData) {
        double score = 0;
        for (String key : inputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += (double) inputData.get(key) * this.weight.get(key);
            }
        }
        score += this.intercept;

        return score;
    }

    @Override
    public abstract Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams);
}
