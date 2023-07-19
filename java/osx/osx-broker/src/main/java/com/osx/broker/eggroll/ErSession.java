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
package com.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static com.osx.core.config.MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_IP;
import static com.osx.core.config.MetaInfo.PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT;

public class ErSession {

    Logger logger = LoggerFactory.getLogger(ErSession.class);
    SessionStatus status = SessionStatus.NEW;
    String sessionId;
    String tag = "";
    String name = "";
    boolean createIfNotExists = true;
    List<ErProcessor> processors = Lists.newArrayList();
    Map<String, String> options = Maps.newHashMap();
    ErSessionMeta erSessionMeta;
    ClusterManagerClient clusterManagerClient;
    List<ErProcessor> rollsBuffer;
    Map<Long, List<ErProcessor>> eggBuffer = Maps.newHashMap();

    public ErSession(String sessionId, boolean createIfNotExists) {

        this.sessionId = sessionId;
        this.createIfNotExists = createIfNotExists;
        clusterManagerClient = new ClusterManagerClient(new CommandClient(new ErEndpoint(PROPERTY_EGGROLL_CLUSTER_MANANGER_IP, PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT.intValue())));
        ErSessionMeta erSessionMetaArgs = new ErSessionMeta();

        erSessionMetaArgs.setId(sessionId);
        erSessionMetaArgs.setName(name);
        erSessionMetaArgs.setStatus(status.name());
        erSessionMetaArgs.setTag(tag);
        erSessionMetaArgs.setProcessors(this.processors);
        erSessionMetaArgs.setOptions(options);
        logger.info("create ErSession ============{}", erSessionMetaArgs);
        if (createIfNotExists) {
            if (processors.isEmpty()) {
                erSessionMeta = clusterManagerClient.getOrCreateSession(erSessionMetaArgs);
            } else {


                erSessionMeta = clusterManagerClient.registerSession(erSessionMetaArgs);
            }
        } else {
            erSessionMeta = clusterManagerClient.getSession(erSessionMetaArgs);

        }

        logger.info("===============dddddd=============={} ", erSessionMeta);

        processors = erSessionMeta.getProcessors();
        status = SessionStatus.valueOf(erSessionMeta.getStatus());
        //            processors.foreach(p => {
//        val processorType = p.processorType
//        if (processorType.toLowerCase().startsWith("egg_")) {
//            eggs_buffer.getOrElseUpdate(p.serverNodeId, ArrayBuffer[ErProcessor]()) += p
//        } else if (processorType.toLowerCase().startsWith("roll_")) {
//            rolls_buffer += p
//        } else {
//            throw new IllegalArgumentException(s"processor type ${processorType} not supported in roll pair")
//        }
//    })

        processors.forEach((processor -> {
            if (processor.getProcessorType().toLowerCase().startsWith("egg_")) {

                if (eggBuffer.get(processor.getServerNodeId()) != null) {
                    eggBuffer.get(processor.getServerNodeId()).add(processor);
                } else {
                    List list = Lists.newArrayList(processor);
                    eggBuffer.put(processor.getServerNodeId(), list);
                }
                ;
            } else if (processor.getProcessorType().toLowerCase().startsWith("roll_")) {
                rollsBuffer.add(processor);
            } else {
                throw new IllegalArgumentException("processor type ${processorType} not supported in roll pair");
            }
        }));


    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public SessionStatus getStatus() {
        return status;
    }

    public void setStatus(SessionStatus status) {
        this.status = status;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isCreateIfNotExists() {
        return createIfNotExists;
    }

    public void setCreateIfNotExists(boolean createIfNotExists) {
        this.createIfNotExists = createIfNotExists;
    }

    public List<ErProcessor> getProcessors() {
        return processors;
    }

    public void setProcessors(List<ErProcessor> processors) {
        this.processors = processors;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    public ErSessionMeta getErSessionMeta() {
        return erSessionMeta;
    }

    public void setErSessionMeta(ErSessionMeta erSessionMeta) {
        this.erSessionMeta = erSessionMeta;
    }

    public ClusterManagerClient getClusterManagerClient() {
        return clusterManagerClient;
    }

    public void setClusterManagerClient(ClusterManagerClient clusterManagerClient) {
        this.clusterManagerClient = clusterManagerClient;
    }

    public List<ErProcessor> getRollsBuffer() {
        return rollsBuffer;
    }

    public void setRollsBuffer(List<ErProcessor> rollsBuffer) {
        this.rollsBuffer = rollsBuffer;
    }

    public Map<Long, List<ErProcessor>> getEggBuffer() {
        return eggBuffer;
    }

    public void setEggBuffer(Map<Long, List<ErProcessor>> eggBuffer) {
        this.eggBuffer = eggBuffer;
    }
    //    private val rolls_buffer = ArrayBuffer[ErProcessor]()
//    private val eggs_buffer = mutable.Map[Long, ArrayBuffer[ErProcessor]]()
    //    def routeToEgg(partition: ErPartition): ErProcessor = {
//        val serverNodeId = partition.processor.serverNodeId
//        val eggCountOnServerNode = eggs(serverNodeId).length
//        val eggIdx = partition.id / eggs.size % eggCountOnServerNode
//
//        eggs(serverNodeId)(eggIdx)
//    }

    public ErProcessor routeToEgg(ErPartition erPartition) {
        long getServerNodeId = erPartition.processor.getServerNodeId();
        if (eggBuffer.get(getServerNodeId) != null) {

            int eggCountOnServerNode = eggBuffer.get(getServerNodeId).size();
            int eggIdx = erPartition.id / eggBuffer.size() % eggCountOnServerNode;
            return eggBuffer.get(getServerNodeId).get(eggIdx);
        }
        return null;
    }


//    private var status = SessionStatus.NEW
//    val clusterManagerClient = new ClusterManagerClient(options)
//    private var sessionMetaArg = ErSessionMeta(
//            id = sessionId,
//            name=name,
//            status = status,
//            tag=tag,
//            processors=processors,
//            options=options)
//    val sessionMeta: ErSessionMeta =
//            if (createIfNotExists) {
//        if (processors.isEmpty) clusterManagerClient.getOrCreateSession(sessionMetaArg)
//        else clusterManagerClient.registerSession(sessionMetaArg)
//    } else {
//        clusterManagerClient.getSession(sessionMetaArg)
//    }
//    processors = sessionMeta.processors
//            status = sessionMeta.status
//
//    private val rolls_buffer = ArrayBuffer[ErProcessor]()
//    private val eggs_buffer = mutable.Map[Long, ArrayBuffer[ErProcessor]]()
//            processors.foreach(p => {
//        val processorType = p.processorType
//        if (processorType.toLowerCase().startsWith("egg_")) {
//            eggs_buffer.getOrElseUpdate(p.serverNodeId, ArrayBuffer[ErProcessor]()) += p
//        } else if (processorType.toLowerCase().startsWith("roll_")) {
//            rolls_buffer += p
//        } else {
//            throw new IllegalArgumentException(s"processor type ${processorType} not supported in roll pair")
//        }
//    })
//
//    val rolls = rolls_buffer.toArray
//    val eggs : Map[Long, Array[ErProcessor]] = eggs_buffer.map(n => (n._1, n._2.toArray)).toMap
//
//    def routeToEgg(partition: ErPartition): ErProcessor = {
//        val serverNodeId = partition.processor.serverNodeId
//        val eggCountOnServerNode = eggs(serverNodeId).length
//        val eggIdx = partition.id / eggs.size % eggCountOnServerNode
//
//        eggs(serverNodeId)(eggIdx)
//    }
}