package com.webank.ai.fate.common.mlmodel.manager;

import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.storage.kv.CurrentProcessKVPool;

public class ModelPool extends CurrentProcessKVPool<String, MLModel> {
    public void ModelPool(){
        System.out.println("model dataPool init");
    }

    public static void main(String[] args){
        System.out.println("test");
    }
}
