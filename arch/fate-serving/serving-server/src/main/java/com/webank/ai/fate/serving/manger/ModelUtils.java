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

import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto;
import com.webank.ai.fate.core.storage.dtable.DTable;
import com.webank.ai.fate.core.storage.dtable.DTableFactory;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import com.webank.ai.fate.core.utils.SceneUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelUtils {
    private static final Logger LOGGER = LogManager.getLogger();
    public static Map<String, byte[]> readModel(String name, String namespace){
        LOGGER.info("read model, name: {} namespace: {}", name, namespace);
        DTable dataTable = DTableFactory.getDTable(name, namespace, 1);
        return dataTable.collect();
    }

    public static PipelineTask loadModel(String name, String namespace){
        Map<String, byte[]> modelBytes = readModel(name, namespace);
        if (modelBytes == null || modelBytes.size() == 0){
            return null;
        }
        PipelineTask pipelineTask = new PipelineTask();
        pipelineTask.initModel(modelBytes);
        return pipelineTask;
    }

    public static String genModelKey(String role, int partyId, Map<String, List<Integer>> allParty, String name, String namespace){
        return StringUtils.join(Arrays.asList(role, partyId, SceneUtils.joinAllParty(allParty), name, namespace), "_");
    }

    public static String genPartnerModelIndexKey(int partnerPartyId, String partnerModelName, String partnerModelNamespace){
        return StringUtils.join(Arrays.asList(partnerPartyId, partnerModelName, partnerModelNamespace), "_");
    }

    public static String[] splitModelKey(String key){
        return StringUtils.split(key, "-");
    }


    public static Map<String, List<Integer>> getAllParty(Map<String, ModelServiceProto.Party> allPartyProto){
        Map<String, List<Integer>> allParty = new HashMap<>();
        allPartyProto.forEach((roleName, party) -> {
            allParty.put(roleName, party.getPartyIdList());
        });
        return allParty;
    }

    public static Map<String, Map<Integer, ModelInfo>> getAllPartyModel(Map<String, ModelServiceProto.RoleModelInfo> allPartyModelProto){
        Map<String, Map<Integer, ModelInfo>> allPartyModel = new HashMap<>();
        allPartyModelProto.forEach((roleName, roleModelInfo)->{
            allPartyModel.put(roleName, new HashMap<>());
            roleModelInfo.getRoleModelInfoMap().forEach((partyId, modelInfo)->{
                allPartyModel.get(roleName).put(partyId, new ModelInfo(modelInfo.getTableName(), modelInfo.getNamespace()));
            });
        });
        return allPartyModel;
    }
}
