///*
// * Copyright 2019 The FATE Authors. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.fedai.osx.broker.router;
//
//import com.fasterxml.jackson.core.type.TypeReference;
//import com.google.common.base.Preconditions;
//import com.google.common.collect.Lists;
//import com.google.common.collect.Maps;
//import com.google.inject.Singleton;
//import org.apache.commons.lang3.StringUtils;
//import org.fedai.osx.broker.util.TelnetUtil;
//import org.fedai.osx.core.config.MetaInfo;
//import org.fedai.osx.core.constant.Dict;
//import org.fedai.osx.core.context.OsxContext;
//import org.fedai.osx.core.context.Protocol;
//import org.fedai.osx.core.datasource.FileRefreshableDataSource;
//import org.fedai.osx.core.exceptions.*;
//import org.fedai.osx.core.flow.PropertyListener;
//import org.fedai.osx.core.frame.Lifecycle;
//import org.fedai.osx.core.frame.ServiceThread;
//import org.fedai.osx.core.router.RouterInfo;
//import org.fedai.osx.core.service.ApplicationStartedRunner;
//import org.fedai.osx.core.service.InboundPackage;
//import org.fedai.osx.core.utils.FileUtils;
//import org.fedai.osx.core.utils.JsonUtil;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.File;
//import java.io.FileNotFoundException;
//import java.io.IOException;
//import java.util.*;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.regex.Matcher;
//import java.util.regex.Pattern;
//
//import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHECK_INTERVAL;
//
//@Singleton
//public class DefaultFateRouterServiceImpl implements RouterService, Lifecycle, ApplicationStartedRunner {
//
//    private static final String IP = "ip";
//    private static final String PORT = "port";
//    private static final String URL = "url";
//    private static final String USE_SSL = "useSSL";
//    private static final String HOSTNAME = "hostname";
//    private static final String negotiationType = "negotiationType";
//    private static final String certChainFile = "certChainFile";
//    private static final String privateKeyFile = "privateKeyFile";
//    private static final String caFile = "caFile";
//    private static final String DEFAULT = "default";
//    private static final String SELF_PARTY="self_party";
//    private static final String ROUTE_TABLE = "route_table";
//    private static final String VERSION = "version";
//    Logger logger = LoggerFactory.getLogger(DefaultFateRouterServiceImpl.class);
//    Pattern urlIpPortPattern = Pattern.compile("((http|ftp|https)://)((([a-zA-Z0-9._-]+)|([0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}))(([a-zA-Z]{2,6})|(:[0-9]{1,4})?))");
//
//    Map<String, Map<String, List<RouterInfo>>> endPointMap = new ConcurrentHashMap<>();
//    Map totalConfig;
//
//    FileRefreshableDataSource fileRefreshableDataSource;
//
//    private  Map parseRouterInfoToMap(RouterInfo  routerInfo){
//        Map  content =  JsonUtil.object2Objcet(routerInfo,Map.class);
//        return  content;
//    }
//
//
//    @Override
//    public synchronized String addRouterInfo(RouterInfo routerInfo) {
//        validateRouterInfo(routerInfo);
//        String desPartyId =  routerInfo.getDesPartyId();
//        String roleId =  routerInfo.getDesRole();
//        Preconditions.checkArgument(StringUtils.isNotEmpty(desPartyId),"des party id is null");
//        if(this.endPointMap.containsKey(desPartyId)){
//            Map roleMap = this.endPointMap.get(desPartyId);
//            if(StringUtils.isNotEmpty(roleId)){
//                roleMap.put(roleId,Lists.newArrayList(routerInfo));
//            }else{
//                roleMap.put(DEFAULT,Lists.newArrayList(routerInfo));
//            }
//        }else{
//            Map newElem = new HashMap<String,List<Map>>();
//            if(StringUtils.isEmpty(roleId)){
//                newElem.put(DEFAULT, Lists.newArrayList(routerInfo));
//            }else{
//                newElem.put(roleId,Lists.newArrayList(routerInfo));
//            }
//            this.endPointMap.putIfAbsent(desPartyId,newElem);
//        }
//        totalConfig.put(ROUTE_TABLE,this.endPointMap);
//        String content = JsonUtil.object2Json(totalConfig);
//        this.saveRouterTable(content);
//        return  content;
//    }
//
//    @Override
//    public void setRouterTable(String content) {
//        if(JsonUtil.validateJson(content)){
//            Map tempConf = JsonUtil.json2Object(content, Map.class);
//            validateAllRouterTable(tempConf);
//            this.saveRouterTable(content);
//        }else {
//            throw new ParameterException("invalid json");
//        }
//    }
//
//    @Override
//    public String getRouterTable() {
//        return JsonUtil.formatJson(JsonUtil.object2Json(totalConfig));
//    }
//
//    public void setSelfPartyIds(Set<String> selfPartyIds){
//        totalConfig.put(SELF_PARTY,selfPartyIds);
//        this.saveRouterTable(JsonUtil.object2Json(totalConfig));
//    }
//
//
//    private RouterInfo buildRouterInfo(RouterInfo endpoint, String srcPartyId, String srcRole, String dstPartyId, String desRole) {
//        Preconditions.checkArgument(endpoint != null);
//        RouterInfo routerInfo = new RouterInfo();
//        if (endpoint.getHost()!= null) {
//            routerInfo.setHost(endpoint.getHost().toString());
//        }
//        if (endpoint.getPort() != null) {
//            routerInfo.setPort(endpoint.getPort());
//        }
//        routerInfo.setDesPartyId(dstPartyId);
//        routerInfo.setSourcePartyId(srcPartyId);
//        routerInfo.setDesRole(desRole);
//        Protocol protocol = Protocol.grpc;
//        if (endpoint.getProtocol() != null) {
//            routerInfo.setProtocol(protocol);
//        }
//        routerInfo.setProtocol(protocol);
//        routerInfo.setUrl(endpoint.getUrl());
//        routerInfo.setUseSSL(endpoint.isUseSSL());
//        routerInfo.setCaFile(endpoint.getCaFile());
//        routerInfo.setCertChainFile(endpoint.getCertChainFile());
//        routerInfo.setPrivateKeyFile(endpoint.getPrivateKeyFile());
//        routerInfo.setKeyStoreFilePath(endpoint.getKeyStoreFilePath());
//        routerInfo.setKeyStorePassword(endpoint.getKeyStorePassword());
//        routerInfo.setTrustStoreFilePath(endpoint.getTrustStoreFilePath());
//        routerInfo.setTrustStorePassword(endpoint.getTrustStorePassword());
//
//        if (routerInfo.getProtocol().equals(Protocol.http)) {
//            if (StringUtils.isEmpty(routerInfo.getUrl())) {
//                throw new InvalidRouteInfoException();
//            }
//        }
//
//        return routerInfo;
//    }
//
//    public RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole) {
//        RouterInfo routerInfo = null;
//        Preconditions.checkArgument(StringUtils.isNotEmpty(dstPartyId), "des party id is null");
//        Map<String, List<RouterInfo>> partyIdMap = this.endPointMap.containsKey(dstPartyId) ? this.endPointMap.get(dstPartyId) : this.endPointMap.get(DEFAULT);
//        if (partyIdMap != null) {
//            if (StringUtils.isNotEmpty(desRole) && partyIdMap.get(desRole) != null) {
//                List<RouterInfo> ips = partyIdMap.getOrDefault(desRole, null);
//                if (ips != null && ips.size() > 0) {
//                    RouterInfo endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
//                    routerInfo = buildRouterInfo(endpoint, srcPartyId, srcRole, dstPartyId, desRole);
//                }
//            } else {
//
//                List<Map> ips = partyIdMap.getOrDefault(DEFAULT, null);
//                if (ips != null && ips.size() > 0) {
//                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
//                    routerInfo = buildRouterInfo(endpoint, srcPartyId, srcRole, dstPartyId, desRole);
//                }
//                if (StringUtils.isNotEmpty(desRole)) {
//                    //    logger.warn("role {} is not found,return default router info ",desRole);
//                }
//            }
//        }
//
//        return routerInfo;
//    }
//
//
//    Map<String, Map<String, List<Map>>> initRouteTable(Map confJson) {
//        Map<String, Map<String, List<Map>>> newRouteTable = new ConcurrentHashMap<>();
//        confJson.forEach((k, v) -> {
//            String coordinatorKey = k.toString();
//            Map coordinatorValue = (Map) v;
//
//            Map<String, List<Map>> serviceTable = newRouteTable.get(coordinatorKey);
//            if (serviceTable == null) {
//                serviceTable = new ConcurrentHashMap<>(4);
//                newRouteTable.put(coordinatorKey, serviceTable);
//            }
//            for (Object roleEntryObject : coordinatorValue.entrySet()) {
//                Map.Entry roleEntry = (Map.Entry) roleEntryObject;
//                String roleKey = roleEntry.getKey().toString();
//                if (roleKey.equals("createTime") || roleKey.equals("updateTime")) {
//                    continue;
//                }
//                List roleValue = (List) roleEntry.getValue();
//                List<Map> endpoints = serviceTable.get(roleKey);
//                if (endpoints == null) {
//                    endpoints = new ArrayList<>();
//                    serviceTable.put(roleKey, endpoints);
//                }
//                for (Object endpointElement : roleValue) {
//                    Map element = Maps.newHashMap();
//                    Map endpointJson = (Map) endpointElement;
//                    element.putAll(endpointJson);
//                    endpoints.add(element);
//                }
//            }
//
//        });
//
//        return newRouteTable;
//    }
//
//    @Override
//    public void init() {
//
//    }
//
//    public void start() {
//        String currentPath = getRouterTablePath();
//        logger.info("load router file {}", currentPath);
//        File confFile = new File(currentPath);
//        fileRefreshableDataSource = null;
//        try {
//            fileRefreshableDataSource = new FileRefreshableDataSource(confFile, (source) -> {
//                return source;
//            });
//            fileRefreshableDataSource.getProperty().addListener(new RouterTableListener());
//
//        } catch (FileNotFoundException e) {
//            logger.error("router file {} is not found", currentPath);
//        }
//        /**
//         * 检查路由表中是否存在回环,是否能连通
//         */
//        ServiceThread routerInfoChecker = new ServiceThread() {
//
//            @Override
//            public void run() {
//                while (true) {
//                    //Map<String, List<Map>> partyIdMap = this.endPointMap.get(dstPartyId);
//                    endPointMap.forEach((desPartyId, desPoint) -> {
//                                desPoint.forEach((role, routerElementMap) -> {
//                                    routerElementMap.forEach(endPoint -> {
//
//                                                String ip = null;
//                                                int port = 0;
//                                                Protocol protocol = Protocol.grpc;
//                                                try {
//                                                    if (endPoint.getProtocol() != null) {
//                                                        try {
//                                                            protocol = endPoint.getProtocol();
//                                                        } catch (Exception e) {
//                                                            logger.warn("route info {}->{} protocol is invalid , please check route_table.json", desPartyId, role);
//                                                        }
//                                                    }
//                                                    ;
//                                                    if (StringUtils.isNotEmpty(endPoint.getUrl() )) {
//                                                        String ipPortString = getIpInfoFromUrl(endPoint.getUrl());
//                                                        if (StringUtils.isNotEmpty(ipPortString)) {
//                                                            ip = ipPortString.split(Dict.COLON)[0];
//                                                            String portString = ipPortString.split(Dict.COLON)[1];
//                                                            port = Integer.parseInt(portString);
//                                                        }
//                                                    }
//                                                    if (protocol.equals(Protocol.grpc)) {
//                                                        if (endPoint.getHost() != null) {
//                                                            ip = endPoint.get(IP).toString();
//                                                        }
//                                                        if (endPoint.get(PORT) != null) {
//                                                            port = ((Number) endPoint.get(PORT)).intValue();
//                                                        }
//                                                    }
//                                                    //if (!MetaInfo.PROPERTY_SELF_PARTY.contains(desPartyId)) {
//
//                                                    boolean isCycle = checkCycle(ip, port);
//                                                    if (isCycle) {
//                                                        logger.warn("route info {}->{}->{}->{} is a cycle , please check route_table.json", desPartyId, role, ip, port);
//                                                    }
//                                                    //endPoint.put(Dict.IS_CYCLE, isCycle);
//                                                    //}
//                                                    checkConnected(desPartyId, role, ip, port);
//
//                                                } catch (Exception ignore) {
//                                                    ignore.printStackTrace();
//                                                }
//                                            }
//                                    );
//                                });
//                            }
//                    );
//
//                    this.waitForRunning(PROPERTY_ROUTER_CHECK_INTERVAL);
//                }
//            }
//
//            @Override
//            public String getServiceName() {
//                return "cycle_checker";
//            }
//        };
//        routerInfoChecker.start();
//    }
//
//    private String getRouterTablePath() {
//        return MetaInfo.PROPERTY_CONFIG_DIR + "/broker/route_table.json";
//    }
//
//    @Override
//    public void destroy() {
//
//    }
//
//    private void checkConnected(String partyId, String role, String ip, int port) {
//
//        if (MetaInfo.PROPERTY_USE_REMOTE_HEALTH_CHECK) {
//            if (StringUtils.isNotEmpty(ip)) {
//
//                boolean result = TelnetUtil.tryTelnet(ip, port);
//                if (!result) {
//                    logger.warn("route info {}->{}->{}->{} unable to connect  , please check route_table.json", partyId, role, ip, port);
//
//                }
//            }
//        }
//    }
//
//    private boolean checkCycle(String ip, int port) {
//
//        boolean cycle = false;
//
//        if (MetaInfo.PROPERTY_OPEN_ROUTE_CYCLE_CHECKER) {
//            String localIp = MetaInfo.INSTANCE_ID.split("_")[0];
//            if (localIp.equals(ip) || Dict.LOCALHOST.equals(ip) || Dict.LOCALHOST2.equals(ip)) {
//                if (MetaInfo.PROPERTY_GRPC_PORT == (port)) {
//                    cycle = true;
//                }
//                if (MetaInfo.PROPERTY_OPEN_GRPC_TLS_SERVER) {
//                    if (MetaInfo.PROPERTY_GRPC_TLS_PORT == port) {
//                        cycle = true;
//                    }
//                }
//                if (MetaInfo.PROPERTY_OPEN_HTTP_SERVER) {
//                    if (MetaInfo.PROPERTY_HTTP_PORT == (port)) {
//                        cycle = true;
//                    }
//                }
//            }
//        }
//        return cycle;
//    }
//
//    @Override
//    public void run(String[] args) throws Exception {
//        this.start();
//    }
//
//    public String getIpInfoFromUrl(String url) {
//        Matcher m = urlIpPortPattern.matcher(url);
//        String result = "";
//        if (m.find()) {
//            result = m.group(3);
//        }
//        return result;
//    }
//
//    public synchronized boolean saveRouterTable( String  content) {
//        try {
//            String routerTablePath = getRouterTablePath();
//            File routerTableFile = new File(routerTablePath);
//            if (!routerTableFile.exists()) {
//                if (!routerTableFile.getParentFile().exists()) {
//                    if (!routerTableFile.getParentFile().mkdirs()) {
//                        logger.warn("mkdir failed : {}", routerTableFile.getParent());
//                        return false;
//                    }
//                }
//                if (!routerTableFile.createNewFile()) {
//                    logger.warn("create router_table.json failed  : {}", routerTableFile.getAbsoluteFile());
//                    return false;
//                }
//            }
//            return FileUtils.writeStr2ReplaceFileSync(JsonUtil.formatJson(content), routerTablePath);
//        } catch (Exception e) {
//            logger.error("save router table failed ", e);
//            return false;
//        }
//    }
//    private void loadSelfParty(Map totalConfig){
//        List selfParties = (List)totalConfig.get(SELF_PARTY);
//        logger.info("load self party {}",selfParties);
//        if(selfParties!=null){
//            Set<String> partySet = new HashSet<>();
//            selfParties.forEach(party->{
//                partySet.add(party.toString());
//            });
//            MetaInfo.PROPERTY_SELF_PARTY = partySet;
//        }else{
//            logger.error("self_party is not found in route_table.json");
//        }
//    }
//
//
//    private void validateRouterInfo(RouterInfo  routerInfo){
//        Preconditions.checkArgument(routerInfo!=null);
//        String desPartyId =  routerInfo.getDesPartyId();
//        Preconditions.checkArgument(StringUtils.isNotEmpty(desPartyId),"des party id is null");
//        if(routerInfo.getProtocol()!=null||Protocol.grpc.equals(routerInfo.getProtocol())){
//            Preconditions.checkArgument(StringUtils.isNotEmpty(routerInfo.getHost()), "route_table.json "+desPartyId+" host/ip is null");
//            Preconditions.checkArgument(routerInfo.getPort()!=null, "route_table.json "+desPartyId+" port is null");
//        }
//    }
//
//
//    private  void  validateAllRouterTable( Map tempConf){
//        if(tempConf==null){
//            throw new SysException("please check route_table.json, it is not a valid json or file is not found");
//        }
//        Object  selfPartyObject = tempConf.get(SELF_PARTY);
//        if(selfPartyObject==null){
//            logger.error("{} is not found in route_table.json",SELF_PARTY);
//            throw new SysException("self_party is not found in route_table.json");
//        }
//        if(!(selfPartyObject instanceof List)){
//            throw new SysException("self_party in route_table.json is invalid, it should be an array");
//        }
//        Map content = (Map) tempConf.get(ROUTE_TABLE);
//        Map<String, Map<String, List<Map>>> temp = initRouteTable(content);
//
//        temp.forEach((k,v)->{
//            if(StringUtils.isEmpty(k)){
//                throw new SysException("");
//            }
//            if(!(v instanceof Map)){
//                throw new SysException("");
//            }
//            v.forEach((role,routerMaps)->{
//                for (Map routerMap : routerMaps) {
//                    RouterInfo routerInfo = buildRouterInfo(routerMap, "", "", k, role);
//                    validateRouterInfo(routerInfo);
//                }
//            });
//        });
//
//
//
//    }
//
//    private void loadRouterTable(String  conf){
//        Map tempConf = JsonUtil.json2Object(conf, Map.class);
//
//        validateAllRouterTable(tempConf);
//        if (tempConf != null) {
//            loadSelfParty(tempConf);
//            Map content = (Map) tempConf.get(ROUTE_TABLE);
//            endPointMap = initRouteTable(content);
//            logger.info("load router table {}", JsonUtil.formatJson(JsonUtil.object2Json(endPointMap)));
//        } else {
//            logger.error("content of route_table.json is invalid , content is {}", conf);
//        }
//
//        totalConfig= tempConf;
//    }
//
//
//    private class RouterTableListener implements PropertyListener<String> {
//
//        @Override
//        public void configUpdate(String value) {
//            logger.warn("found router_table.json has been changed, reload " );
//            loadRouterTable(value);
//        }
//
//        @Override
//        public void configLoad(String value) {
//            loadRouterTable(value);
//        }
//    }
//
//}
