package com.webank.ai.fate.serving.federatedml.transform;

import com.webank.ai.fate.core.mlmodel.buffer.DataTransformServerProto;

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

    public HashMap<String, Object> fit(HashMap<String, Object> inputData, DataTransformServerProto.DataTransformServer dataTransformServer) {
        // missing fill
        boolean isMissingFill = dataTransformServer.getMissingFill();
        if (isMissingFill) {
            ArrayList<String> missingValues = (ArrayList<String>) dataTransformServer.getMissingValueList();
            Map<String, String> missingReplaceValues = dataTransformServer.getMissingReplaceValueMap();
            inputData = replace(inputData, missingValues, missingReplaceValues);
        }

        //outlier replace
        boolean isOutlierReplace = dataTransformServer.getOutlierReplace();
        if (isOutlierReplace) {
            ArrayList<String> outlierValues = (ArrayList<String>) dataTransformServer.getOutlierValueList();
            Map<String, String> outlierReplaceValues = dataTransformServer.getOutlierReplaceValueMap();
            inputData = replace(inputData, outlierValues, outlierReplaceValues);
        }

        //scale
        boolean isScale = dataTransformServer.getIsScale();
        if (isScale) {
            String scaleMethod = dataTransformServer.getScaleMethod();
            if (scaleMethod.equals("MinMaxScale")) {
                MinMaxScale minMaxScale = new MinMaxScale();
                inputData = minMaxScale.fit(inputData, dataTransformServer.getScaleReplaceValueMap());
            }
            else if (scaleMethod.equals("StandardScale")) {
                StandardScale standardScale = new StandardScale();
                inputData = standardScale.fit(inputData, dataTransformServer.getScaleReplaceValueMap());
            }
        }
        return inputData;
    }
}
