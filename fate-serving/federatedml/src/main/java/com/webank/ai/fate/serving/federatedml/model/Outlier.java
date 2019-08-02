package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.OutlierMetaProto.OutlierMeta;
import com.webank.ai.fate.core.mlmodel.buffer.OutlierParamProto.OutlierParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.Map;

public class Outlier extends BaseModel {
    private OutlierMeta outlierMetaProto;
    private OutlierParam outlierParamProto;
    private boolean isOutlier;
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        LOGGER.info("start init Outlier class");
        try {
            this.outlierMetaProto = OutlierMeta.parseFrom(protoMeta);
            this.outlierParamProto = OutlierParam.parseFrom(protoParam);
            this.isOutlier = this.outlierMetaProto.getIsOutlier();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        LOGGER.info("Finish init Outlier class");
        return StatusCode.OK;
    }

    @Override
    public Map<String, Object> predict(Context context, Map<String, Object> inputData, Map<String, Object> predictParams) {
        if (this.isOutlier) {
            List<String> outlierValues = this.outlierMetaProto.getOutlierValueList();
            Map<String, String> outlierReplaceValues = this.outlierParamProto.getOutlierReplaceValueMap();

            LOGGER.info("outlierValues:{}", outlierValues);
            LOGGER.info("outlierReplaceValues:{}", outlierReplaceValues);

            for (String key : inputData.keySet()) {
                String value = inputData.get(key).toString();
                if (outlierValues.contains(value.toLowerCase())) {
                    try {
                        LOGGER.info("value:{}", value);
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
