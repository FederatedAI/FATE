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

package com.webank.ai.fate.serving.utils;

import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.core.storage.dtable.DTableInfo;
import com.webank.ai.fate.core.utils.SceneUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Map;

public class DTableUtils {
    private static final Logger LOGGER = LogManager.getLogger();

    public static DTableInfo genTableInfo(String tableName, String namespace, String role, String partyId, FederatedRoles federatedRoles, String dataType) {

        if (StringUtils.isEmpty(namespace)) {
            namespace = getSceneNamespace(SceneUtils.genSceneKey(role, partyId, federatedRoles), dataType);
        }
        if (StringUtils.isEmpty(tableName)) {
            Map<String, String> versionInfo = VersionControl.getVersionInfo(namespace, "", "", "master");
            if (versionInfo != null) {
                tableName = versionInfo.get("commitId");
            }
        }
        return new DTableInfo(tableName, namespace);
    }

    public static String getSceneNamespace(String sceneKey, String dataType) {
        return StringUtils.join(Arrays.asList(sceneKey, dataType), "_");
    }
}
