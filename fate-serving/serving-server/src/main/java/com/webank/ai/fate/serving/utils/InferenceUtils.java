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

import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.FederatedRoles;

import java.util.*;

import com.webank.ai.fate.core.utils.FederatedUtils;
import com.webank.ai.fate.serving.bean.FederatedInferenceType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class InferenceUtils {
    private static final Logger auditLogger = LogManager.getLogger("audit");

    public static String generateCaseid() {
        return UUID.randomUUID().toString().replace("-", "");
    }

    public static String generateSeqno() {
        return UUID.randomUUID().toString().replace("-", "");
    }

    public static void logInferenceAudited(Enum<FederatedInferenceType> inferenceType, int sceneid, FederatedParty federatedParty, FederatedRoles federatedRoles, String caseid, int statusCode, boolean charge) {
        String inCharge;
        if (charge) {
            inCharge = "1";
        } else {
            inCharge = "0";
        }
        auditLogger.info(" {} {} {} {} {} {} {} {}", inferenceType, sceneid, federatedParty.getRole(), federatedParty.getPartyId(), FederatedUtils.federatedRolesIdentificationString(federatedRoles), caseid, statusCode, inCharge);
    }
}
