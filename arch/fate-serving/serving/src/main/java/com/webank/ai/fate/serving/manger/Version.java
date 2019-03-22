package com.webank.ai.fate.serving.manger;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.webank.ai.fate.core.model.Bytes;
import com.webank.ai.fate.core.storage.kv.StandaloneDTable;
import org.apache.commons.lang3.StringUtils;
import com.webank.ai.fate.core.storage.kv.DTable;

import java.util.HashMap;
import java.util.Map;

public class Version {
    public static String getSceneKey(String sceneId, String myPartyId, String partnerPartyId, String myRole){
        String[] tmp = {sceneId, myPartyId, partnerPartyId, myRole};
        return StringUtils.join(tmp, "_");
    }

    public static DTable getDTable(String nameSpace, String sceneKey, String commitId, String tag, String branch) throws Exception{
        DTable versionTable = new StandaloneDTable();
        versionTable.init(sceneKey, "model_version", 0);
        if (StringUtils.isEmpty(commitId)){
            String currentBranchCommit = new String(versionTable.get("master"));
            commitId = currentBranchCommit;
        }
        String infoJson = new String(versionTable.get(commitId));
        Map<String, Object> dataTableInfo = new ObjectMapper().readValue(infoJson, HashMap.class);
        DTable dataTable = new StandaloneDTable();
        dataTable.init((String)dataTableInfo.get("tableName"), (String)dataTableInfo.get("tableNameSpace"), 0);
        return dataTable;
    }

    public static void main(String[] args) throws Exception{
        String sceneKey = getSceneKey("50000", "9999", "10000", "host");
        DTable dataTable = getDTable("model_data", sceneKey, "", "", "master");
        System.out.println(dataTable.get("model_meta"));
    }
}
