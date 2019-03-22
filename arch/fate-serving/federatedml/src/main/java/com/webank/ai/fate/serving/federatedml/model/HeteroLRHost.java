package com.webank.ai.fate.serving.federatedml.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;

public class HeteroLRHost extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();
    @Override
    public HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams) {
        HashMap<String, Object> newInputData = data_transform(inputData);

        HashMap<String, Object> result = new HashMap<>();
        float score = forward(newInputData);
        result.put("score", score);

        return result;
    }
}
