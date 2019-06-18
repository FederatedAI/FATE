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

package com.webank.ai.fate.core.utils;

import org.apache.commons.lang3.StringUtils;

import java.util.*;

import com.webank.ai.fate.core.bean.FederatedRoles;

public class SceneUtils {
    private static final String sceneKeySeparator = "#";

    public static String genSceneKey(String role, String partyId, FederatedRoles federatedRoles) {
        return StringUtils.join(Arrays.asList(role, partyId, FederatedUtils.federatedRolesIdentificationString(federatedRoles)), sceneKeySeparator);
    }
}
