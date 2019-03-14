package com.webank.ai.fate.common.mlmodel.model;

import java.util.HashMap;

public class HeteroLRHost extends HeteroLR{
    @Override
    public HashMap<String, Object> predict(float[] inputData, HashMap<String, Object> predictParams){
        HashMap<String, Object> result = new HashMap<>();
        float score = 0;
        for(int i=0;i<this.weight.length;i++){
            if (inputData.length>i){
                score += this.weight[i] * inputData[i];
            }
        }
        result.put("score", score);
        return result;
    }
}
