package com.webank.ai.fate.serving.federatedml.model;

import com.webank.ai.fate.core.mlmodel.model.BaseModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.result.StatusCode;
import java.util.HashMap;
import java.util.Map;

public abstract class HeteroLR extends BaseModel<ProtoModelBuffer, HashMap<String, Object>, HashMap<String, Object>> {
    protected Map<String, Float> weight;
    protected float intercept = 0;
    protected Map<String, Object> meta;

    @Override
    public int initModel(ProtoModelBuffer modelBuffer) {
        this.weight = modelBuffer.getParam().getWeightMap();
        this.intercept = modelBuffer.getParam().getIntercept();
        return StatusCode.OK;
    }

    protected HashMap<String, Object> data_transform(HashMap<String, Object> inputData) {
        //TODO
        //get data transform information
        return inputData;
    }

    @Override
    public abstract HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams);
}
