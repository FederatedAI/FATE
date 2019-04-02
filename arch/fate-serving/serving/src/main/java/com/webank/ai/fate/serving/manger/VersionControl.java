/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.serving.manger;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.webank.ai.fate.core.storage.dtable.DTableFactory;
import com.webank.ai.fate.core.utils.Configuration;
import org.apache.commons.lang3.StringUtils;
import com.webank.ai.fate.core.storage.dtable.DTable;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class VersionControl {
    private static final Logger LOGGER = LogManager.getLogger();

    public static String getSceneKey(String sceneId, String myPartyId, String partnerPartyId, String myRole){
        String[] tmp = {sceneId, myPartyId, partnerPartyId, myRole};
        return StringUtils.join(tmp, "_");
    }

    public static DTable getVersionTable(String nameSpace, String sceneId, String partnerPartyId, String myRole){
        String sceneKey = getSceneKey(sceneId, Configuration.getProperty("partyId"), partnerPartyId, myRole);
        DTable versionTable = DTableFactory.getDTable();
        versionTable.init(sceneKey, "model_version", 0);
        return versionTable;
    }

    public static DTable dTableForRead(String nameSpace, String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws Exception{
        DTable versionTable = getVersionTable(nameSpace, sceneId, partnerPartyId, myRole);
        if (versionTable == null){
            return null;
        }
        if (StringUtils.isEmpty(commitId)){
            commitId = new String(versionTable.get(Optional.ofNullable("branch").orElse("master")));
        }
        byte[] infoJson = versionTable.get(commitId);
        if (infoJson==null || infoJson.length < 1){
            return null;
        }
        Map<String, Object> dataTableInfo = new ObjectMapper().readValue(new String(infoJson), HashMap.class);
        DTable dataTable = DTableFactory.getDTable();
        dataTable.init((String)dataTableInfo.get("tableName"), (String)dataTableInfo.get("tableNameSpace"), 0);
        return dataTable;
    }
}
