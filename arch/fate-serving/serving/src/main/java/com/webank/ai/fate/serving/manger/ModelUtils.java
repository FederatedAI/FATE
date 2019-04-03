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

import com.webank.ai.fate.core.storage.dtable.DTable;
import com.webank.ai.fate.core.storage.dtable.DTableFactory;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.Map;

public class ModelUtils {
    public static Map<String, byte[]> readModel(String name, String namespace){
        DTable dataTable = DTableFactory.getDTable(name, namespace, 1);
        return dataTable.collect();
    }

    public static PipelineTask loadModel(String name, String namespace){
        Map<String, byte[]> modelBytes = readModel(name, namespace);
        if (modelBytes == null){
            return null;
        }
        PipelineTask pipelineTask = new PipelineTask();
        pipelineTask.initModel(modelBytes);
        return pipelineTask;
    }

    public static String genModelKey(String name, String namespace){
        return StringUtils.join(Arrays.asList(name, namespace), "_");
    }

    public static String genPartnerModelIndexKey(int partnerPartyId, String partnerModelName, String partnerModelNamespace){
        return StringUtils.join(Arrays.asList(partnerPartyId, partnerModelName, partnerModelNamespace), "_");
    }

    public static String[] splitModelKey(String key){
        return StringUtils.split(key, "-");
    }
}
