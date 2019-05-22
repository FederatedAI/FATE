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

import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.serving.federatedml.PipelineTask;

public class ModelNamespaceData {
    private String namespace;
    private FederatedParty local;
    private FederatedRoles role;
    private String usedModelName;
    private PipelineTask usedModel;

    public ModelNamespaceData() {
        this.namespace = "all";
        this.local = new FederatedParty();
        this.role = new FederatedRoles();
        this.usedModel = new PipelineTask();
    }

    public ModelNamespaceData(String namespace, FederatedParty local, FederatedRoles role, String usedModelName, PipelineTask usedModel) {
        this.namespace = namespace;
        this.local = local;
        this.role = role;
        this.usedModelName = usedModelName;
        this.usedModel = usedModel;
    }

    public void setNamespace(String namespace) {
        this.namespace = namespace;
    }

    public void setLocal(FederatedParty local) {
        this.local = local;
    }

    public void setRole(FederatedRoles role) {
        this.role = role;
    }

    public void setUsedModelName(String usedModelName) {
        this.usedModelName = usedModelName;
    }

    public void setUsedModel(PipelineTask usedModel) {
        this.usedModel = usedModel;
    }

    public String getNamespace() {
        return namespace;
    }

    public FederatedParty getLocal() {
        return local;
    }

    public FederatedRoles getRole() {
        return role;
    }

    public String getUsedModelName() {
        return usedModelName;
    }

    public PipelineTask getUsedModel() {
        return usedModel;
    }
}
