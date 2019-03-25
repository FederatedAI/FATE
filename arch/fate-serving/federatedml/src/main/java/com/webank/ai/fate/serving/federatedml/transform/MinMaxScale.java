package com.webank.ai.fate.serving.federatedml.transform;

import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto;

import java.util.HashMap;
import java.util.Map;

public class MinMaxScale {
    public HashMap<String, Object> fit(HashMap<String, Object> inputData, Map<String, DataTransformProto.ScaleObject> scales) {
        for (String key : inputData.keySet()) {
            try {
                DataTransformProto.ScaleObject scale = scales.get(key);

                float value = (float) inputData.get(key);
                if (value > scale.getFeatUpper())
                    value = 1;
                else if (value < scale.getFeatLower())
                    value = 0;
                else {
                    float range = scale.getFeatUpper() - scale.getFeatLower();
                    if (range <= 0) {
                        value = 0;
                    } else {
                        value = (value - scale.getFeatLower()) / range;
                    }
                }

                float out_range = scale.getOutUpper() - scale.getFeatLower();
                value = value * out_range + scale.getOutLower();
                inputData.put(key, value);

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
