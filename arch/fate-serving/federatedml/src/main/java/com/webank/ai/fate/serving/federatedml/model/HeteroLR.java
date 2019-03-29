package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.model.MachineLearningModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformServerProto.DataTransformServer;

import java.util.HashMap;
import java.util.Map;

public abstract class HeteroLR extends MachineLearningModel<ProtoModelBuffer, HashMap<String, Object>, HashMap<String, Object>> {
    protected Map<String, Float> weight;
    protected float intercept = 0;
    DataTransformServer dataTransformServer;

    @Override
    public int initModel(ProtoModelBuffer modelBuffer) {
        this.weight = modelBuffer.getParam().getWeightMap();
        this.intercept = modelBuffer.getParam().getIntercept();
        this.dataTransformServer = modelBuffer.getDataTransformServer();

        return StatusCode.OK;
    }

    protected float forward(HashMap<String, Object> inputData) {
        float score = 0;
        for (String key : inputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += Float.valueOf(inputData.get(key).toString()) * this.weight.get(key);
            }
        }
        score += this.intercept;

        return score;
    }

    @Override
    public abstract HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams);
}
