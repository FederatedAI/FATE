package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HeteroLRHost extends HeteroLR {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override

    public Map<String, Object> predict(Context context,List<Map<String, Object> > inputData, Map<String, Object> predictParams) {


            LOGGER.info("hetero lr host begin to predict");
            HashMap<String, Object> result = new HashMap<>();
            Map<String, Double> ret = forward(inputData);
            result.put("score", ret.get("score"));

            LOGGER.info("hetero lr host predict ends, result is {}", result);

            return result;
        }
    }
