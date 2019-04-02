package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.OutlierMetaProto.OutlierMeta;
import com.webank.ai.fate.core.mlmodel.buffer.OutlierParamProto.OutlierParam;

import java.util.List;
import java.util.Map;

public class Outlier extends BaseModel {
    private OutlierMeta outlierMetaProto;
    private OutlierParam outlierParamProto;
    private boolean isOutlier;

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        try {
            this.outlierMetaProto = OutlierMeta.parseFrom(protoMeta);
            this.outlierParamProto = OutlierParam.parseFrom(protoParam);
            this.isOutlier = this.outlierMetaProto.getIsOutlier();
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        return StatusCode.OK;
    }

    @Override
    public Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams) {
        if (this.isOutlier) {
            List<String> outlierValues = this.outlierMetaProto.getOutlierValueList();
            Map<String, String> outlierReplaceValues = this.outlierParamProto.getOutlierReplaceValueMap();

            for (String key : inputData.keySet()) {
                if (outlierValues.contains(inputData.get(key))) {
                    try {
                        inputData.put(key, outlierReplaceValues.get(key));
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        inputData.put(key, 0.);
                    }
                }
            }
        }
        return inputData;
    }
}
