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

import com.alibaba.fastjson.JSON;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Job extends JobKey {
    private String fName;

    private String fTag;

    private String fInitiatorPartyId;

    private String fStatus;

    private String fCurrentSteps;

    private String fCurrentTasks;

    private Integer fProgress;

    private Long fCreateTime;

    private Long fUpdateTime;

    private Long fStartTime;
    @JsonProperty(defaultValue = "")
    private Long fEndTime;
    @JsonProperty(defaultValue = "")
    private Long fElapsed;

    private String fRunIp;

    public static void main(String[] args) {

        Job job = new Job();

        System.err.println(JSON.toJSONString(job));


    }

    public String getfName() {
        return fName;
    }

    public void setfName(String fName) {
        this.fName = fName == null ? null : fName.trim();
    }

    public String getfTag() {
        return fTag;
    }

    public void setfTag(String fTag) {
        this.fTag = fTag == null ? null : fTag.trim();
    }

    public String getfInitiatorPartyId() {
        return fInitiatorPartyId;
    }

    public void setfInitiatorPartyId(String fInitiatorPartyId) {
        this.fInitiatorPartyId = fInitiatorPartyId == null ? null : fInitiatorPartyId.trim();
    }

    public String getfStatus() {
        return fStatus;
    }

    public void setfStatus(String fStatus) {
        this.fStatus = fStatus == null ? null : fStatus.trim();
    }

    public String getfCurrentSteps() {
        return fCurrentSteps;
    }

    public void setfCurrentSteps(String fCurrentSteps) {
        this.fCurrentSteps = fCurrentSteps == null ? null : fCurrentSteps.trim();
    }

    public String getfCurrentTasks() {
        return fCurrentTasks;
    }

    public void setfCurrentTasks(String fCurrentTasks) {
        this.fCurrentTasks = fCurrentTasks == null ? null : fCurrentTasks.trim();
    }

    public Integer getfProgress() {
        return fProgress;
    }

    public void setfProgress(Integer fProgress) {
        this.fProgress = fProgress;
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

    public String getfRunIp() {
        return fRunIp;
    }

    public void setfRunIp(String fRunIp) {
        this.fRunIp = fRunIp == null ? null : fRunIp.trim();
    }

}