package com.webank.ai.fate.common.mlmodel.model;

import java.util.HashMap;

public class HeteroLRHost extends HeteroLR {
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
        result.put("score", score);

        return result;
    }
}
