package com.webank.ai.fate.serving.manger;

import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.statuscode.ReturnCode;
import com.webank.ai.fate.core.storage.kv.StandaloneDTable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ModelManager {
    private ModelPool onlineModels;
    private ModelPool modelPool;
    private static final Logger LOGGER = LogManager.getLogger();
    private String modelPackage = "com.webank.ai.fate.serving.federatedml.model";

    public ModelManager(){
        this.onlineModels = new ModelPool();
        this.modelPool = new ModelPool();
    }

    public void updatePool(){
        ArrayList<Map<String, String>> modelInfos = this.getAllModelInfo();
        modelInfos.forEach((modelInfo)->{
            this.onlineModels.put(getOnlineModelKey(modelInfo.get("sceneId"), modelInfo.get("partnerPartyId"), modelInfo.get("myRole")),
                    this.loadModel(modelInfo));
        });
    }

    public MLModel getModel(String sceneId, String partnerPartyId, String myRole){
        return this.onlineModels.get(this.getOnlineModelKey(sceneId, partnerPartyId, myRole));
    }

    public ProtoModelBuffer readModel(String name, String nameSpace, String modelId){
        String DTableName = String.format("%s_%s", name, modelId);
        StandaloneDTable standaloneDTable = new StandaloneDTable(DTableName, nameSpace, 0);
        byte[] metaStream = standaloneDTable.get("model_meta");
        byte[] paramStream = standaloneDTable.get("model_param");
        ProtoModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.deserialize(metaStream, paramStream);
        return modelBuffer;
    }


    public MLModel loadModel(Map<String, String> modelInfo){
        try{
            ProtoModelBuffer modelBuffer = this.readModel(modelInfo.get("name"), modelInfo.get("nameSpace"), modelInfo.get("modelId"));
            Class modelClass = Class.forName(this.modelPackage + "." + modelBuffer.getMeta().getName());
            MLModel mlModel = (MLModel)modelClass.getConstructor().newInstance();
            mlModel.setModelInfo(modelInfo);
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

    private String genModelKey(String sceneId, String partnerPartyId, String myRole, String modelId){
        String[] tmp = {sceneId, partnerPartyId, myRole, modelId};
        return StringUtils.join(tmp, "-");
    }

    private ArrayList<Map<String, String>> getAllModelInfo(){
        ArrayList<Map<String, String>> modelInfos = new ArrayList<>();
        // query from mysql by partyId
        HashMap<String, String> tmp = new HashMap<>();
        tmp.put("sceneId", "500001");
        tmp.put("partnerPartyId", "100001");
        tmp.put("myPartyId", "100001");
        tmp.put("myRole", "guest");
        tmp.put("name", "HeteroLRGuest");
        tmp.put("nameSpace", "HeteroLR");
        tmp.put("modelId", "f0a5b7ca48c711e9b5f2acde48001122");
        modelInfos.add((HashMap<String, String>) tmp.clone());

        tmp.put("myRole", "host");
        tmp.put("name", "HeteroLRHost");
        modelInfos.add((HashMap<String, String>) tmp.clone());
        return modelInfos;
    }

    private ArrayList<Map<String, String>> queryModelInfo(String modelId){
        // query from mysql by partyId and modelId
        ArrayList<Map<String, String>> modelInfos = new ArrayList<>();
        // query from mysql by partyId
        HashMap<String, String> tmp = new HashMap<>();
        tmp.put("sceneId", "500001");
        tmp.put("partnerPartyId", "100001");
        tmp.put("partnerPartyName", "DT");
        tmp.put("myPartyId", "100001");
        tmp.put("myPartyName", "YH");
        tmp.put("myRole", "guest");
        tmp.put("name", "HeteroLRGuest");
        tmp.put("nameSpace", "HeteroLR");
        tmp.put("modelId", modelId);
        modelInfos.add((HashMap<String, String>) tmp.clone());

        tmp.put("myRole", "host");
        tmp.put("name", "HeteroLRHost");
        tmp.put("myPartyName", "DT");
        tmp.put("partnerPartyName", "YH");
        modelInfos.add((HashMap<String, String>) tmp.clone());
        return modelInfos;
    }

    public int publishLoadModel(String modelId){
        ArrayList<Map<String, String>> modelInfos = this.queryModelInfo(modelId);
        modelInfos.forEach((modelInfo)->{
            this.modelPool.put(this.genModelKey(modelInfo.get("sceneId"), modelInfo.get("partnerPartyId"), modelInfo.get("myRole"), modelInfo.get("modelId")),
                    this.loadModel(modelInfo));
        });
        return ReturnCode.OK;
    }

    public int publishOnlineModel(String sceneId, String partnerPartyId, String myRole, String modelId){
        MLModel model = this.modelPool.get(this.genModelKey(sceneId, partnerPartyId, myRole, modelId));
        if (model != null){
            this.onlineModels.put(this.getOnlineModelKey(sceneId, partnerPartyId, myRole), model);
            return ReturnCode.OK;
        }
        else{
            return ReturnCode.NOMODEL;
        }
    }
}
