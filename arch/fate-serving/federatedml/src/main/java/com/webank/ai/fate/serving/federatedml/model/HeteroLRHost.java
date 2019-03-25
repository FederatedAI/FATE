package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.serving.federatedml.transform.DataTransform;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;

public class HeteroLRHost extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();
    @Override
    public HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams) {
        DataTransform dataTransform = new DataTransform();
        HashMap<String, Object> newInputData = dataTransform.fit(inputData, this.dataTransformServer);

        HashMap<String, Object> result = new HashMap<>();
        float score = forward(newInputData);
        result.put("score", score);

        return result;
    }
}
