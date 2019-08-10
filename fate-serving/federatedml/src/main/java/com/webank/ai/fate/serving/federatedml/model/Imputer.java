package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.DataIOMetaProto.ImputerMeta;
import com.webank.ai.fate.core.mlmodel.buffer.DataIOParamProto.ImputerParam;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.HashSet;
import java.util.Map;

public class Imputer {
    private static final Logger LOGGER = LogManager.getLogger();
    public HashSet<String> missingValueSet;
    public Map<String, String> missingReplaceValues;

    public Imputer(List<String> missingValues, Map<String, String> missingReplaceValue) {
    	this.missingValueSet = new HashSet<String>(missingValues);
    	this.missingReplaceValues = missingReplaceValue;
    }

    public Map<String, Object> transform(Map<String, Object> inputData) {
		LOGGER.info("start imputer transform task");
        for (String key : inputData.keySet()) {
            String value = inputData.get(key).toString();
            if (this.missingValueSet.contains(value.toLowerCase())) {
                try {
                    inputData.put(key, this.missingReplaceValues.get(key));
                } catch (Exception ex) {
                    ex.printStackTrace();
                    inputData.put(key, 0.);
                }
            }
        }
        return inputData;
    }
}
