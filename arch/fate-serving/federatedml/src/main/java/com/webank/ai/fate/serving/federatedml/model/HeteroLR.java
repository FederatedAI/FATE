package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.model.BaseModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.statuscode.ReturnCode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto.DataTransform;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto.ScaleObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class HeteroLR extends BaseModel<ProtoModelBuffer, HashMap<String, Object>, HashMap<String, Object>> {
    protected Map<String, Float> weight;
    protected float intercept = 0;
    private ProtoModelBuffer modelBuffer;
    protected Map<String, Object> meta;

    @Override
    public int initModel(ProtoModelBuffer modelBuffer) {
        this.weight = modelBuffer.getParam().getWeightMap();
        this.intercept = modelBuffer.getParam().getIntercept();
        this.modelBuffer = modelBuffer;

        return ReturnCode.OK;
    }

    HashMap<String, Object> transform(HashMap<String, Object> inputData, ArrayList<String> values, Map<String, String> replaceValues) {
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

    protected HashMap<String, Object> data_transform(HashMap<String, Object> inputData) {
        DataTransform dataTransform = this.modelBuffer.getDataTransform();
        boolean isMissingFill = dataTransform.getMissingFill();

        if (isMissingFill) {
            ArrayList<String> missingValues = (ArrayList<String>) dataTransform.getMissingValueList();
            Map<String, String> missingReplaceValues = dataTransform.getMissingReplaceValueMap();
            inputData = transform(inputData, missingValues, missingReplaceValues);
        }

        boolean isOutlierReplace = dataTransform.getOutlierReplace();
        if (isOutlierReplace) {
            ArrayList<String> outlierValues = (ArrayList<String>) dataTransform.getOutlierValueList();
            Map<String, String> outlierReplaceValues = dataTransform.getOutlierReplaceValueMap();
            inputData = transform(inputData, outlierValues, outlierReplaceValues);
        }

        boolean isScale = dataTransform.getIsScale();
        if (isScale) {
            String scaleMethod = dataTransform.getScaleMethod();
            if (scaleMethod.equals("MinMaxScale")) {
                Map<String, ScaleObject> scales = dataTransform.getScaleReplaceValue();
                for (String key : inputData.keySet()) {
                    try {
                        ScaleObject scale = scales.get(key);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }

        return inputData;
    }

    protected float forward(HashMap<String, Object> inputData) {
        float score = 0;
        for (String key : inputData.keySet()) {
            if (this.weight.containsKey(key)) {
                score += (float) inputData.get(key) * this.weight.get(key);
            }
        }
        score += this.intercept;

        return score;
    }

    @Override
    public abstract HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams);
}
