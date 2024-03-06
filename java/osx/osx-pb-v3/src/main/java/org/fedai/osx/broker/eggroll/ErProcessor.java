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

import com.webank.eggroll.core.meta.Meta;

import java.util.concurrent.ConcurrentHashMap;

public class ErProcessor extends BaseProto<Meta.Processor> {

    long id = -1;
    long serverNodeId = -1;
    String name = "";
    String processorType = "";
    String status = "";
    ErEndpoint commandEndpoint;
    ErEndpoint transferEndpoint;
    int pid = -1;
    ConcurrentHashMap options;

    public static ErProcessor parseFromPb(Meta.Processor processor) {
        if (processor != null) {
            ErProcessor erProcessor = new ErProcessor();
            erProcessor.setId(processor.getId());
            erProcessor.setServerNodeId(processor.getServerNodeId());
            erProcessor.setName(processor.getName());
            erProcessor.setProcessorType(processor.getProcessorType());
            erProcessor.setCommandEndpoint(ErEndpoint.parseFromPb(processor.getCommandEndpoint()));
            erProcessor.setTransferEndpoint(ErEndpoint.parseFromPb(processor.getTransferEndpoint()));
            erProcessor.setPid(processor.getPid());
            return erProcessor;
        } else {
            return null;
        }
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public long getServerNodeId() {
        return serverNodeId;
    }

    public void setServerNodeId(long serverNodeId) {
        this.serverNodeId = serverNodeId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getProcessorType() {
        return processorType;
    }

    public void setProcessorType(String processorType) {
        this.processorType = processorType;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public ErEndpoint getCommandEndpoint() {
        return commandEndpoint;
    }

    public void setCommandEndpoint(ErEndpoint commandEndpoint) {
        this.commandEndpoint = commandEndpoint;
    }

    public ErEndpoint getTransferEndpoint() {
        return transferEndpoint;
    }

    public void setTransferEndpoint(ErEndpoint transferEndpoint) {
        this.transferEndpoint = transferEndpoint;
    }

    public int getPid() {
        return pid;
    }

    public void setPid(int pid) {
        this.pid = pid;
    }

    public ConcurrentHashMap getOptions() {
        return options;
    }

    public void setOptions(ConcurrentHashMap options) {
        this.options = options;
    }

    public Meta.Processor toProto() {

        return Meta.Processor.newBuilder().
                setId(this.id).
                setServerNodeId(this.serverNodeId).
                setName(this.name)
                .setProcessorType(this.processorType).
                setCommandEndpoint(this.commandEndpoint.toProto()).setTransferEndpoint(this.transferEndpoint.toProto())
                .setPid(this.pid).build();

    }


}


//case class ErProcessor(id: Long = -1,
//                       serverNodeId: Long = -1,
//                       name: String = StringConstants.EMPTY,
//                       processorType: String = StringConstants.EMPTY,
//                       status: String = StringConstants.EMPTY,
//                       commandEndpoint: ErEndpoint = null,
//                       transferEndpoint: ErEndpoint = null,
//                       pid: Int = -1,
//                       options: java.util.Map[String, String] = new ConcurrentHashMap[String, String](),
//        tag: String = StringConstants.EMPTY) extends NetworkingRpcMessage {
//        override def toString: String = {
//        s"<ErProcessor(id=${id}, serverNodeId=${serverNodeId}, name=${name}, processorType=${processorType}, status=${status}, commandEndpoint=${commandEndpoint}, transferEndpoint=${transferEndpoint}, pid=${pid}, options=${options}, tag=${tag}) at ${hashCode().toHexString}>"
//        }
//        }
