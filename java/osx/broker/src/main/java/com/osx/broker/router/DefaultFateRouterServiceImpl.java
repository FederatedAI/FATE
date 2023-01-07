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
package com.osx.broker.router;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.constant.NegotiationType;
import com.osx.core.datasource.FileRefreshableDataSource;
import com.osx.core.flow.PropertyListener;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.JsonUtil;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class DefaultFateRouterServiceImpl implements FateRouterService {

    private static final String IP = "ip";
    private static final String PORT = "port";
    private static final String URL = "url";
    private static final String USE_SSL = "useSSL";
    private static final String HOSTNAME = "hostname";
    private static final String negotiationType = "negotiationType";
    private static final String certChainFile = "certChainFile";
    private static final String privateKeyFile = "privateKeyFile";
    private static final String caFile = "caFile";
    private static final String DEFAULT = "default";
    private static final String VERSION = "version";
    Logger logger = LoggerFactory.getLogger(DefaultFateRouterServiceImpl.class);
    Map<String, List<RouterInfo>> routerInfoMap = new ConcurrentHashMap<String, List<RouterInfo>>();
    Map<String, Map<String, List<Map>>> endPointMap = new ConcurrentHashMap<>();
    FileRefreshableDataSource fileRefreshableDataSource;

    @Override
    public RouterInfo route(Proxy.Packet packet) {
        Preconditions.checkArgument(packet != null);
        RouterInfo routerInfo = null;
        Proxy.Metadata metadata = packet.getHeader();
        Transfer.RollSiteHeader rollSiteHeader = null;
        try {
            rollSiteHeader = Transfer.RollSiteHeader.parseFrom(metadata.getExt());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        String dstPartyId = rollSiteHeader.getDstPartyId();

        if (StringUtils.isEmpty(dstPartyId)) {
            dstPartyId = metadata.getDst().getPartyId();
        }
        dstPartyId = metadata.getDst().getPartyId();
        String desRole = metadata.getDst().getRole();
        String srcRole = metadata.getSrc().getRole();
        String srcPartyId = metadata.getSrc().getPartyId();
        routerInfo = this.route(srcPartyId, srcRole, dstPartyId, desRole);
        //logger.info("query router info {} to {} {} return {}", srcPartyId, dstPartyId, desRole, routerInfo);
        return routerInfo;
    }


    public RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole) {
        RouterInfo routerInfo = null;
        Map<String, List<Map>> partyIdMap = this.endPointMap.get(dstPartyId);
        if (partyIdMap != null) {

            if (StringUtils.isNotEmpty(desRole)&&partyIdMap.get(desRole) != null) {
                List<Map> ips = partyIdMap.getOrDefault(desRole, null);
                if (ips != null && ips.size() > 0) {
                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
                    routerInfo = new RouterInfo();
                    routerInfo.setHost(endpoint.get(IP).toString());
                    routerInfo.setPort(((Number) endpoint.get(PORT)).intValue());
                    routerInfo.setDesPartyId(dstPartyId);
                    routerInfo.setSourcePartyId(srcPartyId);
                    routerInfo.setVersion(endpoint.get(VERSION) != null ? endpoint.get(VERSION).toString() : null);
                    routerInfo.setNegotiationType(endpoint.get(negotiationType)!=null?endpoint.get(negotiationType).toString():"");
                }
            } else {

                List<Map> ips = partyIdMap.getOrDefault(DEFAULT, null);
                if (ips != null && ips.size() > 0) {
                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
                    routerInfo = new RouterInfo();
                    routerInfo.setHost(endpoint.get(IP).toString());
                    routerInfo.setPort(((Number) endpoint.get(PORT)).intValue());
                    routerInfo.setDesPartyId(dstPartyId);
                    routerInfo.setSourcePartyId(srcPartyId);
                    routerInfo.setVersion(endpoint.get(VERSION) != null ? endpoint.get(VERSION).toString() : null);
                    routerInfo.setNegotiationType(endpoint.get(negotiationType)!=null?endpoint.get(negotiationType).toString():"");
                }
                if(StringUtils.isNotEmpty(desRole)){
                    logger.warn("role {} is not found,return default router info ",desRole);
                }
            }
        }
        return routerInfo;
    }



    Map<String, Map<String, List<Map>>> initRouteTable(Map confJson) {
        // BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder();
        Map<String, Map<String, List<Map>>> newRouteTable = new ConcurrentHashMap<>();
        // loop through coordinator

        confJson.forEach((k,v)->{
            String coordinatorKey = k.toString();
            Map coordinatorValue =  (Map)v;

            Map<String, List<Map>> serviceTable = newRouteTable.get(coordinatorKey);
            if (serviceTable == null) {
                serviceTable = new ConcurrentHashMap<>(4);
                newRouteTable.put(coordinatorKey, serviceTable);
            }
            // loop through role in coordinator
            for (Object roleEntryObject : coordinatorValue.entrySet()) {
                Map.Entry roleEntry = (Map.Entry)roleEntryObject;
                String roleKey = roleEntry.getKey().toString();
                if (roleKey.equals("createTime") || roleKey.equals("updateTime")) {
                    continue;
                }
                List roleValue = (List)roleEntry.getValue();

                List<Map> endpoints = serviceTable.get(roleKey);
                if (endpoints == null) {
                    endpoints = new ArrayList<>();
                    serviceTable.put(roleKey, endpoints);
                }

                // loop through endpoints
                for (Object endpointElement : roleValue) {

                    Map element = Maps.newHashMap();

                    Map endpointJson = (Map)endpointElement;

                    if (endpointJson.get(IP)!=null) {
                        String targetIp = endpointJson.get(IP).toString();
                        element.put(IP, targetIp);
                    }

                    if (endpointJson.get(PORT)!=null) {
                        int targetPort = Integer.parseInt(endpointJson.get(PORT).toString());
                        element.put(PORT, targetPort);
                    }
//                    if(endpointJson.has(URL)){
//                        String url = endpointJson.get(URL).getAsString();
//                        endpointBuilder.setUrl(url);
//                    }

                    if (endpointJson.get(USE_SSL)!=null) {
                        boolean targetUseSSL = Boolean.getBoolean(endpointJson.get(USE_SSL).toString());
                        element.put(USE_SSL, targetUseSSL);
                    }

                    if (endpointJson.get(HOSTNAME)!=null) {
                        String targetHostname = endpointJson.get(HOSTNAME).toString();
                        element.put(HOSTNAME, targetHostname);
                    }

                    if (endpointJson.get(negotiationType)!=null) {
                        String targetNegotiationType = endpointJson.get(negotiationType).toString();
                        element.put(negotiationType, targetNegotiationType);
                    }else{
                        element.put(negotiationType, NegotiationType.PLAINTEXT);
                    }

                    if (endpointJson.get(certChainFile)!=null) {
                        String targetCertChainFile = endpointJson.get(certChainFile).toString();
                        element.put(certChainFile, targetCertChainFile);
                    }

                    if (endpointJson.get(privateKeyFile)!=null) {
                        String targetPrivateKeyFile = endpointJson.get(privateKeyFile).toString();
                        element.put(privateKeyFile, targetPrivateKeyFile);
                    }

                    if (endpointJson.get(caFile)!=null) {
                        String targetCaFile = endpointJson.get(caFile).toString();
                        element.put(caFile, targetCaFile);
                    }
                    if (endpointJson.get(VERSION)!=null) {
                        String targetVersion = endpointJson.get(VERSION).toString();
                        element.put(VERSION, targetVersion);
                    }

                    //BasicMeta.Endpoint endpoint = endpointBuilder.build();
                    endpoints.add(element);
                }
            }

        });

        return newRouteTable;
    }

    public void start() {
        String currentPath = Thread.currentThread().getContextClassLoader().getResource("route_table.json").getPath();
        logger.info("load router file {}", currentPath);
        File confFile = new File(currentPath);
        FileRefreshableDataSource fileRefreshableDataSource = null;
        try {
            fileRefreshableDataSource = new FileRefreshableDataSource(confFile, (source) -> {
                logger.info("read route_table {}", source);
                return source;
            });
            fileRefreshableDataSource.getProperty().addListener(new RouterTableListener());

        } catch (FileNotFoundException e) {
            logger.error("router file {} is not found", currentPath);
        }
    }

    private class RouterTableListener implements PropertyListener<String> {

        @Override
        public void configUpdate(String value) {
            // logger.info("fire router table update {}",value);
            Map confJson =  JsonUtil.json2Object(value,Map.class);
            // JsonObject confJson = JsonParser.parseString(value).getAsJsonObject();
            Map content =(Map) confJson.get("route_table");
            endPointMap = initRouteTable(content);
        }

        @Override
        public void configLoad(String value) {

            //   logger.info("fire router table load {}",value);
           Map confJson =  JsonUtil.json2Object(value,Map.class);
           // JsonObject confJson = JsonParser.parseString(value).getAsJsonObject();
            Map content =(Map) confJson.get("route_table");
            endPointMap = initRouteTable(content);
            logger.info("load router config {}", JsonUtil.formatJson(JsonUtil.object2Json(endPointMap)));
        }
    }


}
