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
package org.fedai.osx.broker.eggroll;
import com.google.common.collect.Lists;
import com.webank.eggroll.core.transfer.Transfer;

import java.util.List;
import java.util.Map;

public class ErRollSiteHeader extends BaseProto<Transfer.RollSiteHeader> {

    String rollSiteSessionId;
    String name;
    String tag;
    String srcRole;
    String srcPartyId;
    String dstRole;
    String dstPartyId;
    String dataType;
    Map<String, String> options;
    Integer totalPartitions;
    Integer partitionId;
    Long totalStreams;
    Long totalBatches;
    Long streamSeq;
    Long batchSeq;
    String stage;

    public static ErRollSiteHeader parseFromPb(Transfer.RollSiteHeader rollSiteHeader) {
        if (rollSiteHeader != null) {
            ErRollSiteHeader erRollSiteHeader = new ErRollSiteHeader();
            erRollSiteHeader.rollSiteSessionId = rollSiteHeader.getRollSiteSessionId();
            erRollSiteHeader.name = rollSiteHeader.getName();
            erRollSiteHeader.tag = rollSiteHeader.getTag();
            erRollSiteHeader.srcRole = rollSiteHeader.getSrcRole();
            erRollSiteHeader.srcPartyId = rollSiteHeader.getSrcPartyId();
            erRollSiteHeader.dstRole = rollSiteHeader.getDstRole();
            erRollSiteHeader.dstPartyId = rollSiteHeader.getDstPartyId();
            erRollSiteHeader.dataType = rollSiteHeader.getDataType();
            erRollSiteHeader.options = rollSiteHeader.getOptionsMap();
            erRollSiteHeader.totalPartitions = rollSiteHeader.getTotalPartitions();
            erRollSiteHeader.partitionId = rollSiteHeader.getPartitionId();
            erRollSiteHeader.totalStreams = rollSiteHeader.getTotalStreams();
            erRollSiteHeader.streamSeq = rollSiteHeader.getStreamSeq();
            erRollSiteHeader.batchSeq = rollSiteHeader.getBatchSeq();
            erRollSiteHeader.stage = rollSiteHeader.getStage();
            return erRollSiteHeader;
        } else {
            return null;
        }

    }

    public String getRollSiteSessionId() {
        return rollSiteSessionId;
    }

    public void setRollSiteSessionId(String rollSiteSessionId) {
        this.rollSiteSessionId = rollSiteSessionId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    public String getSrcRole() {
        return srcRole;
    }

    public void setSrcRole(String srcRole) {
        this.srcRole = srcRole;
    }

    public String getSrcPartyId() {
        return srcPartyId;
    }

    public void setSrcPartyId(String srcPartyId) {
        this.srcPartyId = srcPartyId;
    }

    public String getDstRole() {
        return dstRole;
    }

    public void setDstRole(String dstRole) {
        this.dstRole = dstRole;
    }

    public String getDstPartyId() {
        return dstPartyId;
    }

    public void setDstPartyId(String dstPartyId) {
        this.dstPartyId = dstPartyId;
    }

    public String getDataType() {
        return dataType;
    }

    public void setDataType(String dataType) {
        this.dataType = dataType;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    public Integer getTotalPartitions() {
        return totalPartitions;
    }

    public void setTotalPartitions(Integer totalPartitions) {
        this.totalPartitions = totalPartitions;
    }

    public Integer getPartitionId() {
        return partitionId;
    }

    public void setPartitionId(Integer partitionId) {
        this.partitionId = partitionId;
    }

    public Long getTotalStreams() {
        return totalStreams;
    }

    public void setTotalStreams(Long totalStreams) {
        this.totalStreams = totalStreams;
    }

    public Long getTotalBatches() {
        return totalBatches;
    }

    public void setTotalBatches(Long totalBatches) {
        this.totalBatches = totalBatches;
    }

    public Long getStreamSeq() {
        return streamSeq;
    }

    public void setStreamSeq(Long streamSeq) {
        this.streamSeq = streamSeq;
    }

    public Long getBatchSeq() {
        return batchSeq;
    }

    public void setBatchSeq(Long batchSeq) {
        this.batchSeq = batchSeq;
    }

    public String getStage() {
        return stage;
    }

    public void setStage(String stage) {
        this.stage = stage;
    }

    public String getRsKey(String delim, String prefix) {
        List<String> finalArray =
                Lists.newArrayList(prefix, rollSiteSessionId, name, tag, srcRole, srcPartyId, dstRole, dstPartyId);
        return String.join(delim, finalArray);
    }

    @Override
    Transfer.RollSiteHeader toProto() {
        return null;
    }
}
