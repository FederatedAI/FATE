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
package org.fedai.osx.broker.service;


import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import org.fedai.osx.broker.token.DefaultTokenService;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.StreamLimitMode;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.flow.FlowCounterManager;
import org.fedai.osx.core.flow.FlowRule;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.ptp.TargetMethod;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.token.TokenRequest;
import org.fedai.osx.core.token.TokenResult;
import org.fedai.osx.core.token.TokenResultStatus;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

@Singleton
public class TokenApplyService implements Lifecycle {

    static Logger logger = LoggerFactory.getLogger(TokenApplyService.class);
    @Inject
    FlowCounterManager flowCounterManager;

    @Inject
    DefaultTokenService defaultTokenService;

    ScheduledExecutorService scheduledExecutorService = Executors.newSingleThreadScheduledExecutor();
    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub;

    public TokenApplyService() {

    }

    public PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub buildBlockingStub(String address) {
        String[] ipports = address.split(":");
        RouterInfo routerInfo = new RouterInfo();
        routerInfo.setHost(ipports[0]);
        routerInfo.setPort(Integer.parseInt(ipports[1]));
        ManagedChannel channel = GrpcConnectionFactory.createManagedChannel(routerInfo, true);
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub = PrivateTransferProtocolGrpc.newBlockingStub(channel);
        return blockingStub;

    }

    public void applyToken(OsxContext context, String resource, int count) {

        if (MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.LOCAL.name())
                || MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.CLUSTER.name())) {
            TokenResult localTokenResult = tryLocalLimit(resource, count);
            //    logger.info("request token {} count {} result {}", resource, count, localTokenResult);
            /**
             * 集群限流
             */
            if (MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.CLUSTER.name())) {
                /**
                 * 先尝试本地限流,当本地返回通过时再尝试全局限流
                 */
                if (localTokenResult.getStatus() == TokenResultStatus.OK) {
                    tryClusterLimit(resource, count);
                }
            }
            flowCounterManager.pass(resource, count);
        }
    }


    private TokenResult tryLocalLimit(String resource, int count) {
        boolean needLoop = false;
        int tryTime = 0;
        TokenResult tokenResult;
        do {

            tokenResult = defaultTokenService.requestToken(resource, count, true);
            if (tokenResult != null) {
                ++tryTime;
                //logger.info("prepare to apply token {} {} result {}", resource, count,tokenResult);
                switch (tokenResult.getStatus()) {
                    case TokenResultStatus.OK:
                        needLoop = false;
                        break;
                    case TokenResultStatus.SHOULD_WAIT:
                        //不需要再次循环，唤醒之后直接往下就好

                        int sleepMs = tokenResult.getWaitInMs();
                        logger.info("should wait {} ms", sleepMs);
                        try {
                            Thread.sleep(sleepMs);
                        } catch (InterruptedException igore) {

                        }
                        needLoop = false;
                        break;
                    case TokenResultStatus.BLOCKED:
                        try {
                            sleepMs = tokenResult.getWaitInMs();
                            if (sleepMs > 0) {
                                Thread.sleep(sleepMs);
                            }
                            logger.info("should block {} ms try time {}", sleepMs, tryTime);
                        } catch (InterruptedException e) {
                            logger.error("");
                        }
                        needLoop = true;
                        break;
                    case TokenResultStatus.NO_RULE_EXISTS:
                        break;
                    default:
                        logger.error("error token result {}", tokenResult);
                }
            }
        } while (needLoop && tryTime < MetaInfo.PROPERTY_STREAM_LIMIT_MAX_TRY_TIME);

        return tokenResult;

    }

    private void tryClusterLimit(String resource, int count) {


        TokenRequest tokenRequest = new TokenRequest();
        tokenRequest.setResource(resource);
        tokenRequest.setAcquireCount(count);

        Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();

        //  inboundBuilder.putMetadata(Osx.Header.Version.name(), "123");
        inboundBuilder.putMetadata(Osx.Header.TechProviderCode.name(), MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        //  inboundBuilder.putMetadata(Osx.Header.Token.name(), "testToken");
        // inboundBuilder.putMetadata(Osx.Header.SourceNodeID.name(), "9999");
        //  inboundBuilder.putMetadata(Osx.Header.TargetNodeID.name(), "10000");
        //  inboundBuilder.putMetadata(Osx.Header.SourceInstID.name(), "");
        //  inboundBuilder.putMetadata(Osx.Header.TargetInstID.name(), "");
        //inboundBuilder.putMetadata(Osx.Header.SessionID.name(), "testSessionID");
        inboundBuilder.putMetadata(Osx.Metadata.TargetMethod.name(), TargetMethod.APPLY_TOKEN.name());
        //   inboundBuilder.putMetadata(Osx.Metadata.TargetComponentName.name(), "fateflow");
        //   inboundBuilder.putMetadata(Osx.Metadata.SourceComponentName.name(), "");
        inboundBuilder.setPayload(ByteString.copyFrom(JsonUtil.object2Json(tokenRequest).getBytes(StandardCharsets.UTF_8)));

        if (blockingStub == null) {
            blockingStub = buildBlockingStub(MetaInfo.masterInfo.getInstanceId());
        }
        boolean needLoop = false;
        int tryTime = 0;

        do {
            Osx.Outbound outbound = blockingStub.invoke(inboundBuilder.build());
            ByteString payLoad = outbound.getPayload();
            TokenResult tokenResult = null;
            if (payLoad != null) {
                tokenResult = JsonUtil.json2Object(payLoad.toByteArray(), TokenResult.class);
            }

            if (tokenResult != null) {
                ++tryTime;
                //logger.info("prepare to apply token {} {} result {}", resource, count,tokenResult);
                switch (tokenResult.getStatus()) {
                    case TokenResultStatus.OK:
                        needLoop = false;
                        break;
                    case TokenResultStatus.SHOULD_WAIT:
                        //不需要再次循环，唤醒之后直接往下就好

                        int sleepMs = tokenResult.getWaitInMs();
                        logger.info("should wait {} ms", sleepMs);
                        try {
                            Thread.sleep(sleepMs);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        needLoop = false;
                        break;
                    case TokenResultStatus.BLOCKED:
                        try {
                            sleepMs = tokenResult.getWaitInMs();
                            if (sleepMs > 0) {
                                Thread.sleep(sleepMs);
                                logger.info("should block {} ms try time {}", sleepMs, tryTime);
                            }
                        } catch (InterruptedException e) {
                            logger.error("");
                        }
                        needLoop = true;
                        break;
                    case TokenResultStatus.NO_RULE_EXISTS:
                        break;
                    default:
                        logger.error("error token result {}", tokenResult);
                }
            }
        } while (needLoop && tryTime < MetaInfo.PROPERTY_STREAM_LIMIT_MAX_TRY_TIME);
    }


    private Map parseFlowRule(String content) {
        Map temp = JsonUtil.json2Object(content, Map.class);
        Map result = new HashMap();
        temp.forEach((k, v) -> {
            try {
                result.put(k, JsonUtil.json2Object(JsonUtil.object2Json(v), FlowRule.class));
            } catch (Exception e) {
                logger.error("parse flowRule error", e);
            }
        });
        return result;
    }

    @Override
    public void init() {

    }

    @Override
    public void start() {
//        if(MetaInfo.isCluster()&&StreamLimitMode.CLUSTER.name().equals(MetaInfo.PROPERTY_STREAM_LIMIT_MODE)) {
//            scheduledExecutorService.scheduleAtFixedRate(new Runnable() {
//                @Override
//                public void run() {
//
//                    if (blockingStub != null) {
////                    logger.info("start query flow rule");
//                        try {
//                            Firework.QueryFlowRuleRequest queryFlowRuleRequest = Firework.QueryFlowRuleRequest.newBuilder().build();
//                            Firework.QueryFlowRuleResponse response = blockingStub.queryFlowRule(queryFlowRuleRequest);
//                            //  logger.info("query flow rule result {}", response);
//                            if (response != null) {
//                                String result = response.getResult();
//                                if (StringUtils.isNotEmpty(result)) {
//                                     logger.info("query flow rules {}", result);
//                                }
//                            }
//                        } catch (Exception e) {
//                            e.printStackTrace();
//                        }
//                    } else {
//                        // logger.error("cluster-manager service is not found");
//                    }
//                }
//            }, 0, 10000, TimeUnit.MILLISECONDS);
//        }
    }

    @Override
    public void destroy() {

    }

}
