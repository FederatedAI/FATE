package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.LRModelParamProto.LRModelParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
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

    Map<String, Double> forward(Map<String, Object> inputData) {
        double score = 0;
        int modelWeightHitCount = 0;
        int inputDataHitCount = 0;
        int weightNum = this.weight.size();
        int inputFeaturesNum = inputData.size();
        LOGGER.info("model weight number:{}", weightNum);
        LOGGER.info("input data features number:{}", inputFeaturesNum);

        for (String key : inputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += Double.parseDouble(inputData.get(key).toString()) * this.weight.get(key);
                modelWeightHitCount += 1;
                inputDataHitCount += 1;
                LOGGER.debug("key {} weight is {}, value is {}", key, this.weight.get(key), inputData.get(key));
            }
        }
        score += this.intercept;

        double modelWeightHitRate = -1.0;
        double inputDataHitRate = -1.0;
        try {
            modelWeightHitRate = (double) modelWeightHitCount / weightNum;
            inputDataHitRate = (double) inputDataHitCount / inputFeaturesNum;
        }catch (Exception ex){
            ex.printStackTrace();
        }

        LOGGER.info("model weight hit rate:{}", modelWeightHitRate);
        LOGGER.info("input data features hit rate:{}", inputDataHitRate);

        Map<String, Double> ret = new HashMap<>();
        ret.put("score", score);
        ret.put("modelWrightHitRate", modelWeightHitRate);
        ret.put("inputDataHitRate", inputDataHitRate);
        return ret;
    }

    @Override
    public abstract Map<String, Object> predict(Context context, Map<String, Object> inputData, Map<String, Object> predictParams);
}
