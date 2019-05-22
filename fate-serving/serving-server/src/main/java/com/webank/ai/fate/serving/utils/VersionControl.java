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

import com.webank.ai.fate.core.storage.dtable.DTable;
import com.webank.ai.fate.core.storage.dtable.DTableFactory;
import com.webank.ai.fate.core.utils.ObjectTransform;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

public class VersionControl {
    private static final Logger LOGGER = LogManager.getLogger();

    public static Map<String, String> getVersionInfo(String namespace, String commitId, String tag, String branch) {
        DTable versionDTable = getVersionTable(namespace);
        if (!StringUtils.isEmpty(commitId)) {
            return getVersionInfo(versionDTable, namespace, commitId);
        } else if (!StringUtils.isEmpty(branch)) {
            String branchCurrentCommit = getCurrentBranchCommit(versionDTable, branch);
            if (StringUtils.isEmpty(branchCurrentCommit)) {
                return null;
            }
            return getVersionInfo(versionDTable, namespace, branchCurrentCommit);
        } else {
            return null;
        }

    }

    public static Map<String, String> getVersionInfo(String namespace, String commitId) {
        DTable versionDTable = getVersionTable(namespace);
        return getVersionInfo(versionDTable, namespace, commitId);
    }

    public static Map<String, String> getVersionInfo(DTable versionDTable, String namespace, String commitId) {
        byte[] tmp = versionDTable.get(commitId);
        if (tmp == null) {
            return null;
        }
        return (Map<String, String>) ObjectTransform.json2Bean(new String(tmp), HashMap.class);
    }

    public static DTable getVersionTable(String dataNamespace) {
        DTable dataTable = DTableFactory.getDTable(dataNamespace, "version_control", 1);
        return dataTable;
    }

    public static String getCurrentBranchCommit(DTable versionDTable, String branchName) {
        byte[] tmp = versionDTable.get(branchName);
        if (tmp == null) {
            return null;
        }
        return new String(tmp);
    }
}
