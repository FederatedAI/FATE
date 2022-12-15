package com.osx.broker.router;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.context.Context;
import com.osx.core.router.RouterInfo;
import com.osx.core.datasource.FileRefreshableDataSource;
import com.osx.core.flow.PropertyListener;
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

public class DefaultFateRouterServiceImpl implements FateRouterService  {

    Logger logger = LoggerFactory.getLogger(DefaultFateRouterServiceImpl.class);
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


    Map<String, List<RouterInfo>> routerInfoMap  = new ConcurrentHashMap<String,List<RouterInfo>>();
    Map<String, Map<String, List<Map>>> endPointMap =  new ConcurrentHashMap<>();
    FileRefreshableDataSource fileRefreshableDataSource ;

    @Override
    public RouterInfo route( Proxy.Packet packet) {
        Preconditions.checkArgument(packet!=null);
      //  logger.info("====================== {}",packet);
        RouterInfo  routerInfo = null;
        Proxy.Metadata  metadata = packet.getHeader();
        Transfer.RollSiteHeader rollSiteHeader = null;
        try {
             rollSiteHeader =  Transfer.RollSiteHeader.parseFrom(metadata.getExt());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        String  dstPartyId = rollSiteHeader.getDstPartyId();

        if(StringUtils.isEmpty(dstPartyId)){
            dstPartyId = metadata.getDst().getPartyId();
        }

        dstPartyId = metadata.getDst().getPartyId();
        String desRole = metadata.getDst().getRole();
        String srcRole = metadata.getSrc().getRole();
        String srcPartyId = metadata.getSrc().getPartyId();
        routerInfo = this.route(srcPartyId,srcRole,dstPartyId,desRole);
        logger.info("query router info {} to {} {} return {}",srcPartyId,dstPartyId,desRole,routerInfo);
        return  routerInfo;
    }



    public  RouterInfo route(String srcPartyId,String srcRole,String dstPartyId,String desRole){
        RouterInfo  routerInfo= null;
        Map<String, List<Map>>  partyIdMap = this.endPointMap.get(dstPartyId);
        if(partyIdMap!=null){

            if(partyIdMap.get(desRole)!=null){
                List<Map>  ips  = partyIdMap.getOrDefault(desRole,null);
                if(ips!=null&&ips.size()>0){
                    Map  endpoint = ips.get((int)(System.currentTimeMillis()%ips.size()));
                    routerInfo = new RouterInfo();
                    routerInfo.setHost(endpoint.get(IP).toString());
                    routerInfo.setPort(((Number)endpoint.get(PORT)).intValue());
                    routerInfo.setDesPartyId(dstPartyId);
                    routerInfo.setSourcePartyId(srcPartyId);
                    routerInfo.setVersion(endpoint.get(VERSION)!=null?endpoint.get(VERSION).toString():null);
                }
            }else {
                List<Map> ips = partyIdMap.getOrDefault(DEFAULT, null);
                if (ips != null && ips.size() > 0) {
                    Map endpoint = ips.get((int) (System.currentTimeMillis() % ips.size()));
                    routerInfo = new RouterInfo();
                    routerInfo.setHost(endpoint.get(IP).toString());
                    routerInfo.setPort(((Number)endpoint.get(PORT)).intValue());
                    routerInfo.setDesPartyId(dstPartyId);
                    routerInfo.setSourcePartyId(srcPartyId);
                    routerInfo.setVersion(endpoint.get(VERSION)!=null?endpoint.get(VERSION).toString():null);
                }
            }
        }
     //   logger.info("query router info {} return {}",dstPartyId,routerInfo);
        return  routerInfo;
    }

//    @Override
//    public RouterInfo route(FireworkTransfer.RouteInfo  routeInfo) {
//
//        String desPartyId = routeInfo.getDesPartyId();
//        String srcPartyId = routeInfo.getSrcPartyId();
//        String desRole = routeInfo.getDesRole();
//        String srcRole = routeInfo.getSrcRole();
//        return  this.route(srcRole,srcPartyId,desPartyId,desRole);
//    }



    Map<String, Map<String, List<Map>>>  initRouteTable(JsonObject confJson) {
        // BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder();
        Map<String, Map<String, List<Map>>> newRouteTable = new ConcurrentHashMap<>();
        // loop through coordinator
        for (Map.Entry<String, JsonElement> coordinatorEntry : confJson.entrySet()) {
            String coordinatorKey = coordinatorEntry.getKey();
            JsonObject coordinatorValue = coordinatorEntry.getValue().getAsJsonObject();
          //  logger.info("coordinatorKey {} : {}",coordinatorKey,coordinatorValue);
            Map<String, List<Map>> serviceTable = newRouteTable.get(coordinatorKey);
            if (serviceTable == null) {
                serviceTable = new ConcurrentHashMap<>(4);
                newRouteTable.put(coordinatorKey, serviceTable);
            }
            // loop through role in coordinator
            for (Map.Entry<String, JsonElement> roleEntry : coordinatorValue.entrySet()) {
                String roleKey = roleEntry.getKey();
                if(roleKey.equals("createTime")||roleKey.equals("updateTime")){
                    continue;
                }
                JsonArray roleValue = roleEntry.getValue().getAsJsonArray();

                List<Map> endpoints = serviceTable.get(roleKey);
                if (endpoints == null) {
                    endpoints = new ArrayList<>();
                    serviceTable.put(roleKey, endpoints);
                }

                // loop through endpoints
                for (JsonElement endpointElement : roleValue) {

                    Map  element  = Maps.newHashMap();

                    JsonObject endpointJson = endpointElement.getAsJsonObject();

                    if (endpointJson.has(IP)) {
                        String targetIp = endpointJson.get(IP).getAsString();
                        element.put(IP,targetIp);
                    }

                    if (endpointJson.has(PORT)) {
                        int targetPort = endpointJson.get(PORT).getAsInt();
                        element.put(PORT,targetPort);
                    }
//                    if(endpointJson.has(URL)){
//                        String url = endpointJson.get(URL).getAsString();
//                        endpointBuilder.setUrl(url);
//                    }

                    if (endpointJson.has(USE_SSL)) {
                        boolean targetUseSSL = endpointJson.get(USE_SSL).getAsBoolean();
                        element.put(USE_SSL,targetUseSSL);
                    }

                    if (endpointJson.has(HOSTNAME)) {
                        String targetHostname = endpointJson.get(HOSTNAME).getAsString();
                        element.put(HOSTNAME,targetHostname);
                    }

                    if (endpointJson.has(negotiationType)) {
                        String targetNegotiationType = endpointJson.get(negotiationType).getAsString();
                        element.put(negotiationType,targetNegotiationType);
                    }

                    if (endpointJson.has(certChainFile)) {
                        String targetCertChainFile = endpointJson.get(certChainFile).getAsString();
                        element.put(certChainFile,targetCertChainFile);
                    }

                    if (endpointJson.has(privateKeyFile)) {
                        String targetPrivateKeyFile = endpointJson.get(privateKeyFile).getAsString();
                        element.put(privateKeyFile,targetPrivateKeyFile);
                    }

                    if (endpointJson.has(caFile)) {
                        String targetCaFile = endpointJson.get(caFile).getAsString();
                        element.put(caFile,targetCaFile);
                    }
                    if(endpointJson.has(VERSION)){
                        String targetVersion = endpointJson.get(VERSION).getAsString();
                        element.put(VERSION,targetVersion);
                    }

                    //BasicMeta.Endpoint endpoint = endpointBuilder.build();
                    endpoints.add(element);
                }
            }
        }
        return newRouteTable;
    }




    private  class RouterTableListener implements PropertyListener<String> {

        @Override
        public void configUpdate(String value) {
           // logger.info("fire router table update {}",value);
            JsonObject  confJson = JsonParser.parseString(value).getAsJsonObject();
            JsonObject  content = confJson.get("route_table").getAsJsonObject();
            endPointMap = initRouteTable(content);
        }

        @Override
        public void configLoad(String value) {
         //   logger.info("fire router table load {}",value);
            JsonObject  confJson = JsonParser.parseString(value).getAsJsonObject();
            JsonObject  content = confJson.get("route_table").getAsJsonObject();
            endPointMap = initRouteTable(content);
            logger.info("load router config {}", JsonUtil.formatJson(JsonUtil.object2Json(endPointMap)));
        }
    }

    public void start( ) {
        String currentPath = Thread.currentThread().getContextClassLoader().getResource("route_table.json").getPath();
        logger.info("load router file {}",currentPath);
        File confFile = new File(currentPath);
        FileRefreshableDataSource  fileRefreshableDataSource = null;
        try {
            fileRefreshableDataSource = new FileRefreshableDataSource(confFile,(source)->{
                logger.info("read route_table {}",source);
                return source;
            });
            fileRefreshableDataSource.getProperty().addListener(new RouterTableListener());

        } catch (FileNotFoundException e) {
            logger.error("router file {} is not found",currentPath);
        }
    }


}
