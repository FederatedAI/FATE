package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.DataIOMetaProto.OutlierMeta;
import com.webank.ai.fate.core.mlmodel.buffer.DataIOParamProto.OutlierParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.Map;
import java.util.HashSet;

public class Outlier {
    public HashSet<String> outlierValueSet;
    public Map<String, String> outlierReplaceValues;
    private static final Logger LOGGER = LogManager.getLogger();

    public Outlier(List<String> outlierValues, Map<String, String> outlierReplaceValue) {
        this.outlierValueSet = new HashSet<String>(outlierValues);
        this.outlierReplaceValues = outlierReplaceValue;
    }

    public Map<String, Object> transform(Map<String, Object> inputData) {
		LOGGER.info("start outlier transform task");

        for (String key : inputData.keySet()) {
            String value = inputData.get(key).toString();
            if (this.outlierValueSet.contains(value.toLowerCase())) {
                try {
                    LOGGER.info("value:{}", value);
                    inputData.put(key, outlierReplaceValues.get(key));
                } catch (Exception ex) {
                    ex.printStackTrace();
                    inputData.put(key, 0.);
                }
            }
        }

        return inputData;
    }
}
