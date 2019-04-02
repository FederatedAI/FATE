package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.buffer.ScaleParamProto.StandardScaleParam;

import java.util.Map;

public class StandardScale {
    public Map<String, Object> transform(Map<String, Object> inputData, Map<String, StandardScaleParam> standardScalesMap) {
        for (String key : inputData.keySet()) {
            try {
                StandardScaleParam standardScale = standardScalesMap.get(key);

                double value = (double) inputData.get(key);
                double scale = standardScale.getScale();
                if (scale == 0)
                    scale = 1;

                inputData.put(key, (value - standardScale.getMean() / scale));

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
