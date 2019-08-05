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
package com.webank.ai.fate.board.pojo;

public class JobWithBLOBs extends Job {
    private String fDescription;

    private String fRoles;

    private String fDsl;

    private String fRuntimeConf;

    public String getfDescription() {
        return fDescription;
    }

    public void setfDescription(String fDescription) {
        this.fDescription = fDescription == null ? null : fDescription.trim();
    }

    public String getfRoles() {
        return fRoles;
    }

    public void setfRoles(String fRoles) {
        this.fRoles = fRoles == null ? null : fRoles.trim();
    }

    public String getfDsl() {
        return fDsl;
    }

    public void setfDsl(String fDsl) {
        this.fDsl = fDsl == null ? null : fDsl.trim();
    }

    public String getfRuntimeConf() {
        return fRuntimeConf;
    }

    public void setfRuntimeConf(String fRuntimeConf) {
        this.fRuntimeConf = fRuntimeConf == null ? null : fRuntimeConf.trim();
    }
}