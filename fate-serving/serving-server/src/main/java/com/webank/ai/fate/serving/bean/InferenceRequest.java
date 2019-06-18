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

package com.webank.ai.fate.serving.bean;

import com.webank.ai.fate.serving.utils.InferenceUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.HashMap;
import java.util.Map;

public class InferenceRequest {
    private String appid;
    private String partyId;
    private String modelVersion;
    private String modelId;
    private String seqno;
    private String caseid;
    private Map<String, Object> featureData;

    InferenceRequest() {
        seqno = InferenceUtils.generateSeqno();
        caseid = InferenceUtils.generateCaseid();
        featureData = new HashMap<>();
    }

    public String getSeqno() {
        return seqno;
    }

    public String getAppid() {
        return appid;
    }

    public String getCaseid() {
        return caseid;
    }

    public String getPartyId() {
        return partyId;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public String getModelId() {
        return modelId;
    }

    public Map<String, Object> getFeatureData() {
        return featureData;
    }

    public void setAppid(String appid) {
        this.appid = appid;
        this.partyId = appid;
    }

    public void setPartyId(String partyId) {
        this.partyId = partyId;
        this.appid = partyId;
    }

    public boolean haveAppId() {
        return (!StringUtils.isEmpty(appid) || !StringUtils.isEmpty(partyId));
    }
}
