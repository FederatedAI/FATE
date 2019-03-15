package com.webank.ai.fate.common.mlmodel.manager;

import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.common.storage.kv.StandaloneDTable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ModelManager {
    private ModelPool modelPool;
    private static final Logger LOGGER = LogManager.getLogger();
    private String modelPackage = "com.webank.ai.fate.common.mlmodel.model";

    public ModelManager(){
        this.modelPool = new ModelPool();
    }

    public void updatePool(){
        ArrayList<Map<String, String>> modelInfo = this.getAllModelInfo();
        modelInfo.forEach((item)->{
            this.modelPool.put(getOnlineModelKey(item.get("sceneId"), item.get("partnerPartyId"), item.get("myRole")),
                    this.loadModel(item.get("name"), item.get("nameSpace"), item.get("modelId")));
        });
    }

    public MLModel getModel(String sceneId, String partnerPartyId, String myRole){
        return this.modelPool.get(this.getOnlineModelKey(sceneId, partnerPartyId, myRole));
    }

    public ProtoModelBuffer readModel(String name, String nameSpace, String modelId){
        String DTableName = String.format("%s_%s", name, modelId);
        StandaloneDTable standaloneDTable = new StandaloneDTable(DTableName, nameSpace, 0);
        byte[] metaStream = standaloneDTable.get("meta");
        byte[] paramStream = standaloneDTable.get("param");
        ProtoModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.deserialize(metaStream, paramStream);
        return modelBuffer;
    }


    public MLModel loadModel(String name, String nameSpace, String modelId){
        try{
            ProtoModelBuffer modelBuffer = this.readModel(name, nameSpace, modelId);
            Class modelClass = Class.forName(this.modelPackage + "." + modelBuffer.getMeta().getName());
            MLModel mlModel = (MLModel)modelClass.getConstructor().newInstance();
            mlModel.setModelId(modelId);
            mlModel.initModel(modelBuffer);
            return mlModel;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    private String getOnlineModelKey(String sceneId, String partnerPartyId, String myRole){
        String[] tmp = {sceneId, partnerPartyId, myRole};
        return StringUtils.join(tmp, "-");
    }

    private ArrayList<Map<String, String>> getAllModelInfo(){
        ArrayList<Map<String, String>> modelInfo = new ArrayList<>();
        // query from mysql by partyId
        HashMap<String, String> tmp = new HashMap<>();
        tmp.put("sceneId", "500001");
        tmp.put("partnerPartyId", "100001");
        tmp.put("myPartyId", "100001");
        tmp.put("myRole", "guest");
        tmp.put("name", "HeteroLRGuest");
        tmp.put("nameSpace", "HeteroLR");
        tmp.put("modelId", "2d5374d2471511e9a2e5acde48001122");
        modelInfo.add((HashMap<String, String>) tmp.clone());

        tmp.put("role", "host");
        tmp.put("name", "HeteroLRHost");
        modelInfo.add((HashMap<String, String>) tmp.clone());
        return modelInfo;
    }
}
