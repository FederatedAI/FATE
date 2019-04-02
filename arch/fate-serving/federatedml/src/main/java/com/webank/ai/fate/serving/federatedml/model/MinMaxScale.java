package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.buffer.ScaleParamProto.MinMaxScaleParam;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Map;

public class MinMaxScale {
    private static final Logger LOGGER = LogManager.getLogger();

    public Map<String, Object> transform(Map<String, Object> inputData, Map<String, MinMaxScaleParam> scales) {
        LOGGER.info("Start MinMaxScale transform");
        for (String key : inputData.keySet()) {
            try {
                MinMaxScaleParam scale = scales.get(key);
                double value = Double.parseDouble(inputData.get(key).toString());
                if (value > scale.getFeatUpper())
                    value = 1;
                else if (value < scale.getFeatLower())
                    value = 0;
                else {
                    double range = scale.getFeatUpper() - scale.getFeatLower();
                    if (range <= 0) {
                        value = 0;
                    } else {
                        value = (value - scale.getFeatLower()) / range;
                    }
                }

                double outLower = scale.getOutLower();
                double out_range = scale.getOutUpper() - outLower;
                value = value * out_range + outLower;
                inputData.put(key, value);

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
