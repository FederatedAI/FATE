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

import com.fasterxml.jackson.annotation.JsonProperty;

public class Task {
    private String fTaskId;

    private String fJobId;

    private String fComponentName;

    private String fOperator;

    private String fRunIp;

    private Integer fRunPid;

    private String fStatus;

    private Long fCreateTime;

    private Long fUpdateTime;

    private Long fStartTime;
    @JsonProperty(defaultValue = "")
    private Long fEndTime;

    private Long fElapsed;

    private String fRole;

    private String fPartyId;

    public String getfTaskId() {
        return fTaskId;
    }

    public void setfTaskId(String fTaskId) {
        this.fTaskId = fTaskId == null ? null : fTaskId.trim();
    }

    public String getfJobId() {
        return fJobId;
    }

    public void setfJobId(String fJobId) {
        this.fJobId = fJobId == null ? null : fJobId.trim();
    }

    public String getfComponentName() {
        return fComponentName;
    }

    public void setfComponentName(String fComponentName) {
        this.fComponentName = fComponentName == null ? null : fComponentName.trim();
    }

    public String getfOperator() {
        return fOperator;
    }

    public void setfOperator(String fOperator) {
        this.fOperator = fOperator == null ? null : fOperator.trim();
    }

    public String getfRunIp() {
        return fRunIp;
    }

    public void setfRunIp(String fRunIp) {
        this.fRunIp = fRunIp == null ? null : fRunIp.trim();
    }

    public Integer getfRunPid() {
        return fRunPid;
    }

    public void setfRunPid(Integer fRunPid) {
        this.fRunPid = fRunPid;
    }

    public String getfStatus() {
        return fStatus;
    }

    public void setfStatus(String fStatus) {
        this.fStatus = fStatus == null ? null : fStatus.trim();
    }

    public Long getfCreateTime() {
        return fCreateTime;
    }

    public void setfCreateTime(Long fCreateTime) {
        this.fCreateTime = fCreateTime;
    }

    public Long getfUpdateTime() {
        return fUpdateTime;
    }

    public void setfUpdateTime(Long fUpdateTime) {
        this.fUpdateTime = fUpdateTime;
    }

    public Long getfStartTime() {
        return fStartTime;
    }

    public void setfStartTime(Long fStartTime) {
        this.fStartTime = fStartTime;
    }

    public Long getfEndTime() {
        return fEndTime;
    }

    public void setfEndTime(Long fEndTime) {
        this.fEndTime = fEndTime;
    }

    public Long getfElapsed() {
        return fElapsed;
    }

    public void setfElapsed(Long fElapsed) {
        this.fElapsed = fElapsed;
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