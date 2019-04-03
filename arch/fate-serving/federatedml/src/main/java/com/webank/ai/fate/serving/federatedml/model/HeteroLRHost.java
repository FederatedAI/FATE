package com.webank.ai.fate.serving.federatedml.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class HeteroLRHost extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams) {

        HashMap<String, Object> result = new HashMap<>();
        double score = forward(inputData);
        result.put("score", score);

        return result;
    }
}
