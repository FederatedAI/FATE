package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.ScaleMetaProto.ScaleMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ScaleParamProto.ScaleParam;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.Map;

public class Scaler extends BaseModel {
    private ScaleMeta scaleMeta;
    private ScaleParam scaleParam;
    private boolean need_run;
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public int initModel(byte[] protoMeta, byte[] protoParam) {
        LOGGER.info("start init Scale class");
        try {
            this.scaleMeta = this.parseModel(ScaleMeta.parser(), protoMeta);
            this.scaleParam = this.parseModel(ScaleParam.parser(), protoParam);
            this.need_run = scaleParam.getNeedRun();
        } catch (Exception ex) {
            ex.printStackTrace();
            return StatusCode.ILLEGALDATA;
        }
        LOGGER.info("Finish init Scale class");
        return StatusCode.OK;
    }

    @Override
    public Map<String, Object> predict(Context context, List<Map<String, Object>> inputData, Map<String, Object> predictParams) {
//        if (this.need_run) {
//            String scaleMethod = this.scaleMeta.getMethod();
//            if (scaleMethod.toLowerCase().equals("min_max_scale")) {
//                MinMaxScale minMaxScale = new MinMaxScale();
//                inputData = minMaxScale.transform(inputData, this.scaleParam.getColScaleParamMap());
//            } else if (scaleMethod.toLowerCase().equals("standard_scale")) {
//                StandardScale standardScale = new StandardScale();
//                inputData = standardScale.transform(inputData, this.scaleParam.getColScaleParamMap());
//            }
//        }
//        return inputData;

        return  null;
    }
}
