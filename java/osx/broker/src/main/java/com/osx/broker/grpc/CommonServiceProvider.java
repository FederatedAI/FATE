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
//
//package com.osx.transfer.grpc;
//import com.google.common.base.Preconditions;
//import com.google.common.collect.Maps;
//import com.osx.core.bean.CommonActionType;
//import com.osx.core.context.Context;
//import com.osx.transfer.ServiceContainer;
//import com.webank.ai.fate.api.networking.common.CommonServiceProto;
//import com.google.protobuf.ByteString;
//import org.apache.commons.lang3.StringUtils;
//
//
//import java.lang.reflect.Method;
//import java.util.List;
//import java.util.Map;
//
//
//public class CommonServiceProvider extends AbstractMethodServiceProvider {
//
//
//    public  CommonServiceProvider(){
//        try {
//            this.registerMethod(CommonActionType.LIST_PROPS.name(),
//                    this.getClass().getDeclaredMethod("listProps",Context.class,InboundPackage.class));
//            this.registerMethod(CommonActionType.QUERY_JVM.name(),
//                    this.getClass().getDeclaredMethod("listJvmMem",Context.class,InboundPackage.class));
//            this.registerMethod(CommonActionType.QUERY_METRICS.name(),
//                    this.getClass().getDeclaredMethod("queryMetrics",Context.class,InboundPackage.class));
//        } catch (NoSuchMethodException e) {
//            e.printStackTrace();
//        }
//    }
//
//    @Override
//    protected Object transformExceptionInfo(Context context, ExceptionInfo data) {
//        CommonServiceProto.CommonResponse.Builder builder = CommonServiceProto.CommonResponse.newBuilder();
//        builder.setStatusCode(data.getCode());
//        builder.setMessage(data.getMessage());
//        return builder.build();
//    }
//
//
//    public CommonServiceProto.CommonResponse queryMetrics(Context context, InboundPackage inboundPackage) {
//        CommonServiceProto.QueryMetricRequest queryMetricRequest = (CommonServiceProto.QueryMetricRequest) inboundPackage.getBody();
//        long beginMs = queryMetricRequest.getBeginMs();
//        long endMs = queryMetricRequest.getEndMs();
//        String sourceName = queryMetricRequest.getSource();
//        CommonServiceProto.MetricType type = queryMetricRequest.getType();
//        List<MetricNode> metricNodes = null;
//        if (type.equals(CommonServiceProto.MetricType.INTERFACE)) {
//            if (StringUtils.isBlank(sourceName)) {
//                metricNodes = ServiceContainer.flowCounterManager.queryAllMetrics(beginMs, 300);
//            } else {
//                metricNodes = ServiceContainer.flowCounterManager.queryMetrics(beginMs, endMs, sourceName);
//            }
//        }
//
//        CommonServiceProto.CommonResponse.Builder builder = CommonServiceProto.CommonResponse.newBuilder();
//        String response = metricNodes != null ? JsonUtil.object2Json(metricNodes) : "";
//        builder.setStatusCode(StatusCode.SUCCESS);
//        builder.setData(ByteString.copyFrom(response.getBytes()));
//        return builder.build();
//    }
//
//    @FateServiceMethod(name = "UPDATE_FLOW_RULE")
//    public CommonServiceProto.CommonResponse updateFlowRule(Context context, InboundPackage inboundPackage) {
//        CommonServiceProto.UpdateFlowRuleRequest updateFlowRuleRequest = (CommonServiceProto.UpdateFlowRuleRequest) inboundPackage.getBody();
//        ServiceContainer.flowCounterManager.setAllowQps(updateFlowRuleRequest.getSource(), updateFlowRuleRequest.getAllowQps());
//        CommonServiceProto.CommonResponse.Builder builder = CommonServiceProto.CommonResponse.newBuilder();
//        builder.setStatusCode(StatusCode.SUCCESS);
//        builder.setMessage(Dict.SUCCESS);
//        return builder.build();
//    }
//
//    @FateServiceMethod(name = "LIST_PROPS")
//    public CommonServiceProto.CommonResponse listProps(Context context, InboundPackage inboundPackage) {
//        CommonServiceProto.QueryPropsRequest queryPropsRequest = (CommonServiceProto.QueryPropsRequest) inboundPackage.getBody();
//        String keyword = queryPropsRequest.getKeyword();
//        CommonServiceProto.CommonResponse.Builder builder = CommonServiceProto.CommonResponse.newBuilder();
//        builder.setStatusCode(StatusCode.SUCCESS);
//        Map metaInfoMap = MetaInfo.toMap();
//        Map map;
//        if (StringUtils.isNotBlank(keyword)) {
//            Map resultMap = Maps.newHashMap();
//            metaInfoMap.forEach((k, v) -> {
//                if (String.valueOf(k).toLowerCase().indexOf(keyword.toLowerCase()) > -1) {
//                    resultMap.put(k, v);
//                }
//            });
//            map = resultMap;
//        } else {
//            map = metaInfoMap;
//        }
//        builder.setData(ByteString.copyFrom(JsonUtil.object2Json(map).getBytes()));
//        return builder.build();
//    }
//
//    @FateServiceMethod(name = "QUERY_JVM")
//    public CommonServiceProto.CommonResponse listJvmMem(Context context, InboundPackage inboundPackage) {
//        try {
//            CommonServiceProto.CommonResponse.Builder builder = CommonServiceProto.CommonResponse.newBuilder();
//            builder.setStatusCode(StatusCode.SUCCESS);
//            List<JvmInfo> jvmInfos = JvmInfoCounter.getMemInfos();
//            builder.setData(ByteString.copyFrom(JsonUtil.object2Json(jvmInfos).getBytes()));
//            return builder.build();
//        } catch (Exception e) {
//            throw new SysException(e.getMessage());
//        }
//    }
//
//
//
//    @Override
//    protected void printFlowLog(Context context) {
//        Method method = (Method) this.getMethodMap().get(context.getActionType());
//
//        flowLogger.info("{}|{}|{}|{}|{}|{}|{}|{}",
//
//                context.getCaseId(), context.getReturnCode(), context.getCostTime(),
//                context.getDownstreamCost(), serviceName + "." + method.getName(), context.getRouterInfo() != null ? context.getRouterInfo() : "",
//                MetaInfo.PROPERTY_PRINT_INPUT_DATA ? context.getData(Dict.INPUT_DATA) : "",
//                MetaInfo.PROPERTY_PRINT_OUTPUT_DATA ? context.getData(Dict.OUTPUT_DATA) : ""
//        );
//    }
//
//
//
//
//}
