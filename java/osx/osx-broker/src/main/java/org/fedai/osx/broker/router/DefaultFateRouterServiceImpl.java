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
package org.fedai.osx.broker.router;

import com.fasterxml.jackson.core.type.TypeReference;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.api.constants.Protocol;
import org.fedai.osx.api.context.Context;
import org.fedai.osx.api.router.RouterInfo;
import org.fedai.osx.broker.util.TelnetUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.datasource.FileRefreshableDataSource;
import org.fedai.osx.core.exceptions.CycleRouteInfoException;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.InvalidRouteInfoException;
import org.fedai.osx.core.flow.PropertyListener;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.frame.ServiceThread;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.utils.FileUtils;
import org.fedai.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DefaultFateRouterServiceImpl implements FateRouterService, Lifecycle {

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

    //Pattern urlIpPort = Pattern.compile("(\\d+\\.\\d+\\.\\d+\\.\\d+)\\:(\\d+)");

    Pattern urlIpPortPattern = Pattern.compile("((http|ftp|https)://)((([a-zA-Z0-9._-]+)|([0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}))(([a-zA-Z]{2,6})|(:[0-9]{1,4})?))");

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
        String dstPartyId = null;
        try {
            rollSiteHeader = Transfer.RollSiteHeader.parseFrom(metadata.getExt());
            if (rollSiteHeader != null) {
                dstPartyId = rollSiteHeader.getDstPartyId();
            }
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        if (StringUtils.isEmpty(dstPartyId)) {
            dstPartyId = metadata.getDst().getPartyId();
        }
        String desRole = metadata.getDst().getRole();
        String srcRole = metadata.getSrc().getRole();
        String srcPartyId = metadata.getSrc().getPartyId();
        routerInfo = this.route(srcPartyId, srcRole, dstPartyId, desRole);
        //logger.info("query router info {} to {} {} return {}", srcPartyId, dstPartyId, desRole, routerInfo);
        return routerInfo;
    }

    private RouterInfo buildRouterInfo(Map endpoint, String srcPartyId, String srcRole, String dstPartyId, String desRole) {

        Preconditions.checkArgument(endpoint != null);
        RouterInfo routerInfo = new RouterInfo();
        if (endpoint.get(IP) != null) {
            routerInfo.setHost(endpoint.get(IP).toString());
        }
        if (endpoint.get(PORT) != null) {
            routerInfo.setPort(((Number) endpoint.get(PORT)).intValue());
        }
        routerInfo.setDesPartyId(dstPartyId);
        routerInfo.setSourcePartyId(srcPartyId);
        routerInfo.setVersion(endpoint.get(VERSION) != null ? endpoint.get(VERSION).toString() : null);
        routerInfo.setNegotiationType(endpoint.get(negotiationType) != null ? endpoint.get(negotiationType).toString() : "");
        routerInfo.setDesRole(desRole);
        Protocol protocol = Protocol.grpc;
        if (endpoint.get(Dict.PROTOCOL) != null) {
            try {
                protocol = Protocol.valueOf(endpoint.get(Dict.PROTOCOL).toString());
            } catch (Exception ignore) {

            }
        }
        routerInfo.setProtocol(protocol);
        routerInfo.setUrl(endpoint.get(Dict.URL) != null ? endpoint.get(Dict.URL).toString() : "");
        routerInfo.setUseSSL(endpoint.get(Dict.USE_SSL) != null && Boolean.parseBoolean(endpoint.get(Dict.USE_SSL).toString()));
        routerInfo.setCaFile(endpoint.get(Dict.CA_FILE) != null ? endpoint.get(Dict.CA_FILE).toString() : "");
        routerInfo.setCertChainFile(endpoint.get(Dict.CERT_CHAIN_FILE) != null ? endpoint.get(Dict.CERT_CHAIN_FILE).toString() : "");
        routerInfo.setPrivateKeyFile(endpoint.get(Dict.PRIVATE_KEY_FILE) != null ? endpoint.get(Dict.PRIVATE_KEY_FILE).toString() : "");
        if (routerInfo.getProtocol().equals(Protocol.http)) {
            if (StringUtils.isEmpty(routerInfo.getUrl())) {
                throw new InvalidRouteInfoException();
            }
        }
        if (endpoint.get(Dict.IS_CYCLE) != null && (Boolean) endpoint.get(Dict.IS_CYCLE)) {
            logger.error("router info {} has a cycle invoke", routerInfo.toKey());
            throw new CycleRouteInfoException("router info has a cycle invoke");
        }
        return routerInfo;
    }

    public RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole) {
        // logger.info("try to find routerInfo =={}=={}=={}=={}",srcPartyId,srcRole,dstPartyId,desRole);
        RouterInfo routerInfo = null;
        Map<String, List<Map>> partyIdMap = this.endPointMap.containsKey(dstPartyId)?this.endPointMap.get(dstPartyId):this.endPointMap.get(DEFAULT);
        if (partyIdMap != null) {
            if (StringUtils.isNotEmpty(desRole) && partyIdMap.get(desRole) != null) {
                List<Map> ips = partyIdMap.getOrDefault(desRole, null);
                if (ips != null && ips.size() > 0) {
                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
                    routerInfo = buildRouterInfo(endpoint, srcPartyId, srcRole, dstPartyId, desRole);
                }
            } else {

                List<Map> ips = partyIdMap.getOrDefault(DEFAULT, null);
                if (ips != null && ips.size() > 0) {
                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
                    routerInfo = buildRouterInfo(endpoint, srcPartyId, srcRole, dstPartyId, desRole);
                }
                if (StringUtils.isNotEmpty(desRole)) {
                    //    logger.warn("role {} is not found,return default router info ",desRole);
                }
            }
        }

        return routerInfo;
    }


    Map<String, Map<String, List<Map>>> initRouteTable(Map confJson) {
        // BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder();
        Map<String, Map<String, List<Map>>> newRouteTable = new ConcurrentHashMap<>();
        // loop through coordinator

        confJson.forEach((k, v) -> {
            String coordinatorKey = k.toString();
            Map coordinatorValue = (Map) v;

            Map<String, List<Map>> serviceTable = newRouteTable.get(coordinatorKey);
            if (serviceTable == null) {
                serviceTable = new ConcurrentHashMap<>(4);
                newRouteTable.put(coordinatorKey, serviceTable);
            }
            // loop through role in coordinator
            for (Object roleEntryObject : coordinatorValue.entrySet()) {
                Map.Entry roleEntry = (Map.Entry) roleEntryObject;
                String roleKey = roleEntry.getKey().toString();
                if (roleKey.equals("createTime") || roleKey.equals("updateTime")) {
                    continue;
                }
                List roleValue = (List) roleEntry.getValue();

                List<Map> endpoints = serviceTable.get(roleKey);
                if (endpoints == null) {
                    endpoints = new ArrayList<>();
                    serviceTable.put(roleKey, endpoints);
                }
                // loop through endpoints
                for (Object endpointElement : roleValue) {
                    Map element = Maps.newHashMap();
                    Map endpointJson = (Map) endpointElement;
                    element.putAll(endpointJson);
                    endpoints.add(element);
                }
            }

        });

        return newRouteTable;
    }

    @Override
    public void init() {

    }

    public void start() {
        String currentPath = getRouterTablePath();
        logger.info("load router file {}", currentPath);
        File confFile = new File(currentPath);
        FileRefreshableDataSource fileRefreshableDataSource = null;
        try {
            fileRefreshableDataSource = new FileRefreshableDataSource(confFile, (source) -> {
                //   logger.info("read route_table {}", source);
                return source;
            });
            fileRefreshableDataSource.getProperty().addListener(new RouterTableListener());

        } catch (FileNotFoundException e) {
            logger.error("router file {} is not found", currentPath);
        }
        /**
         * 检查路由表中是否存在回环,是否能连通
         */
        ServiceThread routerInfoChecker = new ServiceThread() {

            @Override
            public void run() {
                while (true) {
                    //Map<String, List<Map>> partyIdMap = this.endPointMap.get(dstPartyId);
                    endPointMap.forEach((desPartyId, desPoint) -> {
                                desPoint.forEach((role, routerElementMap) -> {
                                    routerElementMap.forEach(endPoint -> {

                                                String ip = null;
                                                int port = 0;
                                                Protocol protocol = Protocol.grpc;
                                                try {
                                                    if (endPoint.get(Dict.PROTOCOL) != null) {
                                                        try {
                                                            protocol = Protocol.valueOf(endPoint.get(Dict.PROTOCOL).toString());
                                                        } catch (Exception e) {
                                                            logger.warn("route info {}->{} protocol is invalid , please check route_table.json", desPartyId, role);
                                                        }
                                                    }
                                                    ;
                                                    if (endPoint.get(Dict.URL) != null) {
                                                        String ipPortString = getIpInfoFromUrl(endPoint.get(Dict.URL).toString());
                                                        if (StringUtils.isNotEmpty(ipPortString)) {
                                                            ip = ipPortString.split(Dict.COLON)[0];
                                                            String portString = ipPortString.split(Dict.COLON)[1];
                                                            port = Integer.parseInt(portString);
                                                        }
                                                    }
                                                    if (protocol.equals(Protocol.grpc)) {
                                                        if (endPoint.get(IP) != null) {
                                                            ip = endPoint.get(IP).toString();
                                                        }
                                                        if (endPoint.get(PORT) != null) {
                                                            port = ((Number) endPoint.get(PORT)).intValue();
                                                        }
                                                    }
                                                    //if (!MetaInfo.PROPERTY_SELF_PARTY.contains(desPartyId)) {

                                                    boolean isCycle = checkCycle(ip, port);
                                                    if (isCycle) {
                                                        logger.warn("route info {}->{}->{}->{} is a cycle , please check route_table.json", desPartyId, role, ip, port);
                                                    }
                                                    endPoint.put(Dict.IS_CYCLE, isCycle);
                                                    //}
                                                    checkConnected(desPartyId, role, ip, port);

                                                } catch (Exception ignore) {
                                                    ignore.printStackTrace();
                                                }
                                            }
                                    );
                                });
                            }
                    );

                    this.waitForRunning(60000);
                }
            }

            @Override
            public String getServiceName() {
                return "cycle_checker";
            }
        };
        routerInfoChecker.start();
    }

    private String getRouterTablePath() {
        return MetaInfo.PROPERTY_CONFIG_DIR + "/broker/route_table.json";
    }

    @Override
    public void destroy() {

    }

    private void checkConnected(String partyId, String role, String ip, int port) {

        if (MetaInfo.PROPERTY_USE_REMOTE_HEALTH_CHECK) {
            if (StringUtils.isNotEmpty(ip)) {

                boolean result = TelnetUtil.tryTelnet(ip, port);
                if (!result) {
                    //    logger.warn("route info {}->{}->{}->{} unable to connect  , please check route_table.json", partyId, role, ip, port);

                }
            }
        }
    }

    private boolean checkCycle(String ip, int port) {

        boolean cycle = false;

        if(MetaInfo.PROPERTY_OPEN_ROUTE_CYCLE_CHECKER) {
            String localIp = MetaInfo.INSTANCE_ID.split(":")[0];

            if (localIp.equals(ip) || Dict.LOCALHOST.equals(ip) || Dict.LOCALHOST2.equals(ip)) {
                if (MetaInfo.PROPERTY_GRPC_PORT == (port)) {
                    cycle = true;
                }
                if (MetaInfo.PROPERTY_OPEN_GRPC_TLS_SERVER) {
                    if (MetaInfo.PROPERTY_GRPC_TLS_PORT == port) {
                        cycle = true;
                    }
                }
                if (MetaInfo.PROPERTY_OPEN_HTTP_SERVER) {
                    if (MetaInfo.PROPERTY_HTTP_PORT == (port)) {
                        cycle = true;
                    }
                }
            }
        }

        return cycle;
    }


    private class RouterTableListener implements PropertyListener<String> {

        @Override
        public void configUpdate(String value) {
            logger.info("found router_table.json has been changed, update content {}",value);
            Map confJson = JsonUtil.json2Object(value, Map.class);
            // JsonObject confJson = JsonParser.parseString(value).getAsJsonObject();
            Map content = (Map) confJson.get("route_table");
            endPointMap = initRouteTable(content);
        }

        @Override
        public void configLoad(String value) {
            Map confJson = JsonUtil.json2Object(value, Map.class);
            if(confJson!=null){

               // throw new ConfigErrorException("content of route_table.json is invalid");

                Map content = (Map) confJson.get("route_table");
                endPointMap = initRouteTable(content);
                logger.info("load router config {}", JsonUtil.formatJson(JsonUtil.object2Json(endPointMap)));

            }else{
                logger.error("content of route_table.json is invalid , content is {}",value);

            }
                   }
    }


    public String getIpInfoFromUrl(String url) {
        Matcher m = urlIpPortPattern.matcher(url);
        String result = "";
        if (m.find()) {
            result = m.group(3);
        }
        return result;
    }

    public boolean saveRouterTable(Context context, InboundPackage<Proxy.Packet> data) {
        try {
            String inboundRouteJson = (String) context.getData("route");
            if (StringUtils.isNotBlank(inboundRouteJson)) {
                Map<String, Object> routeMap = JsonUtil.object2Objcet(inboundRouteJson, new TypeReference<Map<String, Object>>() {
                });
                Map<String, Object> route_table = (Map<String, Object>) routeMap.get("route_table");
                route_table.forEach((partyId, value) -> {
                    List<RouterInfo> routeList = (List<RouterInfo>) value;
                    for (RouterInfo routerInfo : routeList) {
                        routerInfo.setProtocol(StringUtils.isBlank(routerInfo.getProtocol().toString()) ? Protocol.grpc : routerInfo.getProtocol());
                    }
                });
                inboundRouteJson = JsonUtil.object2Json(routeMap);
            }
            String routerTablePath = getRouterTablePath();
            File routerTableFile = new File(routerTablePath);
            if (!routerTableFile.exists()) {
                if (!routerTableFile.getParentFile().exists()) {
                    if (!routerTableFile.getParentFile().mkdirs()) {
                        logger.warn("mkdir failed : {}", routerTableFile.getParent());
                        return false;
                    }
                }
                if (!routerTableFile.createNewFile()) {
                    logger.warn("create router_table.json failed  : {}", routerTableFile.getAbsoluteFile());
                    return false;
                }
            }
            return FileUtils.writeStr2ReplaceFileSync(JsonUtil.formatJson(inboundRouteJson), routerTablePath);
        } catch (Exception e) {
            logger.error("save router table failed ", e);
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg("save router table failed");
            return false;
        }
    }

    public static void main(String[] args) {
//        System.out.println(MetaInfo.PROPERTY_USER_DIR);
//        System.out.println(MetaInfo.PROPERTY_USER_HOME);
//        System.out.println(Thread.currentThread().getContextClassLoader().getResource("").getPath());
//        System.out.println(Thread.currentThread().getContextClassLoader().getResource("route_table.json"));
//        System.out.println(Thread.currentThread().getContextClassLoader().getResource("flowRule.json"));
        DefaultFateRouterServiceImpl defaultFateRouterService = new DefaultFateRouterServiceImpl();
        defaultFateRouterService.getIpInfoFromUrl("http://127.0.0.1:9000/xxxx");


    }


}
