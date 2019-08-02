package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.buffer.ScaleParamProto.StandardScaleParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Map;

public class StandardScale {
    private static final Logger LOGGER = LogManager.getLogger();

    public Map<String, Object> transform(Context context, Map<String, Object> inputData, Map<String, StandardScaleParam> standardScalesMap) {
        LOGGER.info("Start StandardScale transform");
        for (String key : inputData.keySet()) {
            try {
                StandardScaleParam standardScale = standardScalesMap.get(key);

                double value = Double.parseDouble(inputData.get(key).toString());
                double scale = standardScale.getScale();
                if (scale == 0)
                    scale = 1;

                value = (value - standardScale.getMean()) / scale;
                inputData.put(key, value);

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
