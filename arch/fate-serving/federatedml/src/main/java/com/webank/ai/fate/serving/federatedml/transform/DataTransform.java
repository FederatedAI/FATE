package com.webank.ai.fate.serving.federatedml.transform;

import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class DataTransform {

    private HashMap<String, Object> replace(HashMap<String, Object> inputData, ArrayList<String> values, Map<String, String> replaceValues) {
        for (String key : inputData.keySet()) {
            if (values.contains(inputData.get(key))) {
                try {
                    inputData.put(key, replaceValues.get(key));
                } catch (Exception ex) {
                    ex.printStackTrace();
                    inputData.put(key, 0.);
                }
            }
        }
        return inputData;
    }

    public HashMap<String, Object> fit(HashMap<String, Object> inputData, DataTransformProto.DataTransform dataTransform) {
        // missing fill
        boolean isMissingFill = dataTransform.getMissingFill();
        if (isMissingFill) {
            ArrayList<String> missingValues = (ArrayList<String>) dataTransform.getMissingValueList();
            Map<String, String> missingReplaceValues = dataTransform.getMissingReplaceValueMap();
            inputData = replace(inputData, missingValues, missingReplaceValues);
        }

        //outlier replace
        boolean isOutlierReplace = dataTransform.getOutlierReplace();
        if (isOutlierReplace) {
            ArrayList<String> outlierValues = (ArrayList<String>) dataTransform.getOutlierValueList();
            Map<String, String> outlierReplaceValues = dataTransform.getOutlierReplaceValueMap();
            inputData = replace(inputData, outlierValues, outlierReplaceValues);
        }

        //scale
        boolean isScale = dataTransform.getIsScale();
        if (isScale) {
            String scaleMethod = dataTransform.getScaleMethod();
            if (scaleMethod.equals("MinMaxScale")) {
                MinMaxScale minMaxScale = new MinMaxScale();
                inputData = minMaxScale.fit(inputData, dataTransform.getScaleReplaceValueMap());
            }
            else if (scaleMethod.equals("StandardScale")) {
                StandardScale standardScale = new StandardScale();
                inputData = standardScale.fit(inputData, dataTransform.getScaleReplaceValueMap());
            }
        }
        return inputData;
    }
}
