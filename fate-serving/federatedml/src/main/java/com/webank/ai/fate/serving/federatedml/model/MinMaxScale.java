package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.buffer.ScaleParamProto.ColumnScaleParam;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Map;

public class MinMaxScale {
    private static final Logger LOGGER = LogManager.getLogger();

    public Map<String, Object> transform(Map<String, Object> inputData, Map<String, ColumnScaleParam> scales) {
        LOGGER.info("Start MinMaxScale transform");
        for (String key : inputData.keySet()) {
            try {
                if (scales.containsKey(key)) {
                    ColumnScaleParam scale = scales.get(key);
                    double value = Double.parseDouble(inputData.get(key).toString());
                    if (value > scale.getColumnUpper())
                        value = 1;
                    else if (value < scale.getColumnLower())
                        value = 0;
                    else {
                        double range = scale.getColumnUpper() - scale.getColumnLower();
                        if (range < 0) {
                            LOGGER.warn("min_max_scale range may be error, it should be larger than 0, but is {}, set value to 0 ", range);
                            value = 0;
                        } else {
                            if (Math.abs(range - 0) < 1e-6) {
                                range = 1;
                            }
                            value = (value - scale.getColumnLower()) / range;
                        }
                    }

                    inputData.put(key, value);
                } else {
                    LOGGER.warn("feature {} is not in scale, maybe missing or do not need to be scaled", key);
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return inputData;
    }
}
