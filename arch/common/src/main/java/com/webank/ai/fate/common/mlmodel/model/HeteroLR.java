package com.webank.ai.fate.common.mlmodel.model;

import com.webank.ai.fate.common.mlmodel.buffer.ModelBuffer;
import com.webank.ai.fate.common.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.common.mlmodel.model.BaseModel;
import com.webank.ai.fate.common.statuscode.ReturnCode;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public abstract class HeteroLR extends BaseModel<ModelBuffer, HashMap<String, Object>, HashMap<String, Object>, HashMap<String, Object>> {
    protected HashMap<String, Float> weight;
    protected float intercept = 0;
    protected HashMap<String, Object> meta;

    public void setWeight(HashMap<String, Float> weight) {
        this.weight = (HashMap<String, Float>) weight.clone();
    }

    public HashMap<String, Float> getWeight() {
        return this.weight;
    }

    @Override
    public ModelBuffer export_model() {
//        ArrayList<Float> weight = new ArrayList<>();
//        Collections.addAll(weight, ArrayUtils.toObject(this.weight));
//        ModelBuffer modelBuffer = new ProtoModelBuffer();
//        modelBuffer.setMetaField("name", this.getClass().getName());
//        modelBuffer.setParamField("weight", weight);
        return new ProtoModelBuffer();
    }

    @Override
    public int init_model(ModelBuffer modelBuffer) {
        this.weight = (HashMap<String, Float>) modelBuffer.getParamField("weight");
        this.intercept = (float) modelBuffer.getParamField("intercept");
        return ReturnCode.OK;
    }

    protected HashMap<String, Object> data_transform(HashMap<String, Object> inputData) {
        //TODO
        //get data transform information
        return inputData;
    }

    @Override
    public abstract HashMap<String, Object> predict(HashMap<String, Object> inputData, HashMap<String, Object> predictParams);
}
