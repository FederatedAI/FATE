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

public class JobKey {
    private String fJobId;

    private String fRole;

    private String fPartyId;

    public String getfJobId() {
        return fJobId;
    }

    public void setfJobId(String fJobId) {
        this.fJobId = fJobId == null ? null : fJobId.trim();
    }

    public String getfRole() {
        return fRole;
    }

    public void setfRole(String fRole) {
        this.fRole = fRole == null ? null : fRole.trim();
    }

    public String getfPartyId() {
        return fPartyId;
    }

    public void setfPartyId(String fPartyId) {
        this.fPartyId = fPartyId == null ? null : fPartyId.trim();
    }
}