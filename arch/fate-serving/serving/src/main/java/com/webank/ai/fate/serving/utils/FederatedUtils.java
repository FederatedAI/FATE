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

import com.webank.ai.fate.api.serving.InferenceServiceProto.FederatedMeta;
import com.webank.ai.fate.core.utils.Configuration;

public class FederatedUtils {
    public static FederatedMeta.Builder genResponseMetaBuilder(FederatedMeta requestMeta){
        FederatedMeta.Builder responseMetaBuilder = FederatedMeta.newBuilder();
        responseMetaBuilder.setSceneId(requestMeta.getSceneId());
        responseMetaBuilder.setMyPartyId(Configuration.getPropertyInt("partyId"));
        responseMetaBuilder.setPartnerPartyId(requestMeta.getMyPartyId());
        responseMetaBuilder.setMyRole(getMyRole(requestMeta.getMyRole()));
        return responseMetaBuilder;
    }

    public static String getMyRole(String requestRole) {
        String myRole;
        switch (requestRole) {
            case "guestUser":
                myRole = "guest";
                break;
            case "guest":
                myRole = "host";
                break;
            default:
                myRole = "unknown";
        }
        return myRole;
    }
}
