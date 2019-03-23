package com.webank.ai.fate.serving.federatedml.transform;

import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto;

import java.util.HashMap;
import java.util.Map;

public class StandardScale {
    public HashMap<String, Object> fit(HashMap<String, Object> inputData, Map<String, DataTransformProto.ScaleObject> scales) {
        for (String key : inputData.keySet()) {
            try {
                DataTransformProto.ScaleObject scale = scales.get(key);

                float value = (float) inputData.get(key);
                float std_var = scale.getScale();
                if (std_var == 0)
                    std_var = 1;

                inputData.put(key, (value - scale.getMean() / std_var));

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
