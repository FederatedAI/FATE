package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.ImputerMetaProto.ImputerMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ImputerParamProto.ImputerParam;

import java.util.List;
import java.util.Map;

public class Imputer extends BaseModel {
    private ImputerMeta imputerMetaProto;
    private ImputerParam imputerParamProto;
    private boolean isImputer;

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        try {
            this.imputerMetaProto = ImputerMeta.parseFrom(protoMeta);
            this.imputerParamProto = ImputerParam.parseFrom(protoParam);
            this.isImputer = imputerMetaProto.getIsImputer();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        return StatusCode.OK;
    }

    @Override
    public Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams) {
        if (this.isImputer) {
            List<String> missingValues = this.imputerMetaProto.getMissingValueList();
            Map<String, String> missingReplaceValues = this.imputerParamProto.getMissingReplaceValueMap();
            for (String key : inputData.keySet()) {
                if (missingValues.contains(inputData.get(key))) {
                    try {
                        inputData.put(key, missingReplaceValues.get(key));
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
