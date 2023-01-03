package com.osx.broker.service;

import com.firework.cluster.rpc.Firework;
import com.firework.cluster.rpc.FireworkServiceGrpc;
import com.osx.broker.ServiceContainer;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.StreamLimitMode;
import com.osx.core.context.Context;
import com.osx.core.flow.FlowRule;
import com.osx.core.frame.Lifecycle;
import com.osx.core.token.TokenResult;
import com.osx.core.token.TokenResultStatus;
import com.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

public class TokenApplyService implements Lifecycle {

    static Logger logger = LoggerFactory.getLogger(TokenApplyService.class);
    ScheduledExecutorService scheduledExecutorService = Executors.newSingleThreadScheduledExecutor();
    private FireworkServiceGrpc.FireworkServiceBlockingStub blockingStub;

    public TokenApplyService() {

    }

    public FireworkServiceGrpc.FireworkServiceBlockingStub buildBlockingStub(String address) {
//        String[] ipports=   address.split(":");
//        ManagedChannel channel = GrpcConnectionFactory.getManagedChannel(ipports[0],Integer.parseInt(ipports[1]));
//        FireworkServiceGrpc.FireworkServiceBlockingStub  blockingStub = FireworkServiceGrpc.newBlockingStub(channel);
//        return  blockingStub;
        return null;
    }

    public void applyToken(Context context, String resource, int count) {

        if (MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.LOCAL.name()) || MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.CLUSTER.name())) {
            TokenResult localTokenResult = tryLocalLimit(resource, count);
            logger.info("request token {} count {} result {}", resource, count, localTokenResult);
            /**
             * 集群限流
             */
            if (MetaInfo.PROPERTY_STREAM_LIMIT_MODE.equals(StreamLimitMode.CLUSTER.name())) {
                /**
                 * 先尝试本地限流,当本地返回通过时再尝试全局限流
                 */
                // FlowRule localRule = rules.get(resource);

                if (localTokenResult.getStatus() == TokenResultStatus.OK) {
                    tryClusterLimit(resource, count);
                }
                //   flowCounterManager.pass(resource, count);
                //        }else{
                ////            logger.info("service {} resource {} has no flow rule",context.getServiceName(),resource);
                //        }

            }
            ServiceContainer.flowCounterManager.pass(resource, count);
        }
    }


    private TokenResult tryLocalLimit(String resource, int count) {
        boolean needLoop = false;
        int tryTime = 0;
        TokenResult tokenResult;
        do {
            tokenResult = ServiceContainer.defaultTokenService.requestToken(resource, count, true);
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
        Firework.TokenRequest.Builder tokenRequestBuilder = Firework.TokenRequest.newBuilder();
        tokenRequestBuilder.setCount(count);
        tokenRequestBuilder.setResource(resource);
        if (blockingStub == null) {
            blockingStub = buildBlockingStub(MetaInfo.masterInfo.getInstanceId());
        }
        boolean needLoop = false;
        int tryTime = 0;

        do {
            Firework.TokenResponse tokenResult = blockingStub.applyToken(tokenRequestBuilder.build());
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
