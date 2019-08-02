package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.ImputerMetaProto.ImputerMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ImputerParamProto.ImputerParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.Map;

public class Imputer extends BaseModel {
    private ImputerMeta imputerMetaProto;
    private ImputerParam imputerParamProto;
    private boolean isImputer;
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        LOGGER.info("start init Imputer class");
        try {
            this.imputerMetaProto = ImputerMeta.parseFrom(protoMeta);
            this.imputerParamProto = ImputerParam.parseFrom(protoParam);
            this.isImputer = imputerMetaProto.getIsImputer();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        LOGGER.info("Finish init Imputer class");
        return StatusCode.OK;
    }

    @Override
    public Map<String, Object> predict(Context context, Map<String, Object> inputData, Map<String, Object> predictParams) {
        if (this.isImputer) {
            List<String> missingValues = this.imputerMetaProto.getMissingValueList();
            Map<String, String> missingReplaceValues = this.imputerParamProto.getMissingReplaceValueMap();
            for (String key : inputData.keySet()) {
                String value = inputData.get(key).toString();
                if (missingValues.contains(value.toLowerCase())) {
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
