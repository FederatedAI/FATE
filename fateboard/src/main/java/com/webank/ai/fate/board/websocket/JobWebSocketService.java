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
package com.webank.ai.fate.board.websocket;

import com.alibaba.fastjson.JSON;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.webank.ai.fate.board.conf.Configurator;
import com.webank.ai.fate.board.controller.JobDetailController;
import com.webank.ai.fate.board.controller.JobManagerController;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.pojo.Job;
import com.webank.ai.fate.board.services.JobManagerService;
import com.webank.ai.fate.board.utils.Dict;
import com.webank.ai.fate.board.utils.HttpClientPool;
import com.webank.ai.fate.board.utils.ThreadPoolTaskExecutorUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;

import javax.websocket.*;
import javax.websocket.server.PathParam;
import javax.websocket.server.ServerEndpoint;
import java.io.IOException;
import java.sql.Time;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;


@ServerEndpoint(value = "/websocket/progress/{jobId}/{role}/{partyId}", configurator = Configurator.class)
@Component
public class JobWebSocketService implements InitializingBean, ApplicationContextAware {

    static HttpClientPool httpClientPool;

    static JobManagerService jobManagerService;

    static ApplicationContext applicationContext;

    static JobDetailController jobDetailController;

    static JobManagerController jobManagerController;

    static Logger logger = LoggerFactory.getLogger(JobWebSocketService.class);

    static ConcurrentHashMap jobSessionMap = new ConcurrentHashMap();

    static ThreadPoolTaskExecutor asyncServiceExecutor;

    static ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();

    static  {
        executorService.scheduleAtFixedRate(()->{
            schedule();
        },1000, 1000,TimeUnit.MILLISECONDS);

    }

    /**
     * call method when building connection
     **/
    @OnOpen
    public void onOpen(Session session, @PathParam("jobId") String jobId, @PathParam("role") String role, @PathParam("partyId") Integer partyId) {

        String jobKey = jobId + ":" + role + ":" + partyId;

        jobSessionMap.put(session, jobKey);

        logger.info("websocket job id {} open ,session size{}", jobKey, jobSessionMap.size());

    }

    /**
     * call method when closing connection
     */
    @OnClose
    public void onClose(Session session) {

        logger.info("websocket session closed");
        jobSessionMap.remove(session);
    }

    /**
     * call method when receiving message from client
     *
     * @param message message from client
     */
    @OnMessage
    public void onMessage(String message, Session session) {


    }

    /**
     * @param session
     * @param error
     */
    @OnError
    public void onError(Session session, Throwable error) {
        logger.error("there is a error!", error);
        try {
            session.close();
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            jobSessionMap.remove(session);
        }
        error.printStackTrace();
    }


    static  void schedule() {
       // logger.info("job schedule begin");
        try {
            Set<Map.Entry> entrySet = jobSessionMap.entrySet();
            if (logger.isDebugEnabled()) {
                logger.debug("job process schedule start,session map size {}", jobSessionMap.size());
            }
            Map<String, Set<Session>> jobMaps = Maps.newHashMap();
            jobSessionMap.forEach((k, v) -> {
                Session session = (Session) k;
                String jobKey = (String) v;
                Set<Session> sessions = jobMaps.get(jobKey);
                if (sessions == null) {
                    sessions = new HashSet<Session>();
                }
                sessions.add(session);
                jobMaps.put(jobKey, sessions);

            });
            if (jobMaps.size() > 0) {
                logger.info("job websocket job size {}", jobMaps.size());
            }
            jobMaps.forEach((k, v) -> {
                        String[] args = k.split(":");
                        Preconditions.checkArgument(args.length == 3);
                        String jobId = args[0];
                        String role = args[1];
                        Integer partyId = new Integer(args[2]);
                        Job job = jobManagerService.queryJobByConditions(args[0], args[1], args[2]);
                        if (job != null) {

                            HashMap<String, Object> flushToWebData = new HashMap<>(16);
                            Integer process = job.getfProgress();
                            long now = System.currentTimeMillis();
                            long duration = 0;
                            Long startTime = job.getfStartTime();
                            Long endTime = job.getfEndTime();
                            if (endTime != null) {
                                duration = endTime - startTime;

                            } else {
                                duration = now - startTime;
                            }

                            String status = job.getfStatus();

                            flushToWebData.put(Dict.JOB_PROCESS, process);
                            flushToWebData.put(Dict.JOB_DURATION, duration);
                            flushToWebData.put(Dict.JOB_STATUS, status);

                            Map param = Maps.newHashMap();
                            param.put(Dict.JOBID, jobId);
                            param.put(Dict.ROLE, role);
                            param.put(Dict.PARTY_ID, partyId);

                            Future<?> dependencyFuture = ThreadPoolTaskExecutorUtil.submitListenable(asyncServiceExecutor, () -> {
                                ResponseResult responseResult = jobDetailController.getDagDependencies(JSON.toJSONString(param));

                                return responseResult;
                            }, new int[]{500}, new int[]{3});

                            try {

                                try {
                                    flushToWebData.put(Dict.DEPENDENCY_DATA, dependencyFuture.get());
                                } catch (Exception e) {
                                    logger.error("DEPENDENCY_DATA ERROR", e);
                                }

                            } catch (Exception e) {

                                logger.error("job websocket error ", e);

                            }
                            v.forEach(session -> {

                                if (session.isOpen()) {
                                    try {
                                        session.getBasicRemote().sendText(JSON.toJSONString(flushToWebData));
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                        logger.error("IOException", e);
                                    }
                                } else {
                                    v.remove(session);

                                    jobSessionMap.remove(session);
                                }

                            });
                        } else {
                            logger.error("job {} is not exist", k);
                        }
                        ;

                    }
            );
        }catch (Exception e ){

        }

    }

    @Override
    public void afterPropertiesSet() throws Exception {
        jobManagerService = (JobManagerService) applicationContext.getBean("jobManagerService");
        httpClientPool = (HttpClientPool) applicationContext.getBean("httpClientPool");
        jobDetailController = (JobDetailController) applicationContext.getBean("jobDetailController");
        asyncServiceExecutor = (ThreadPoolTaskExecutor) applicationContext.getBean("asyncServiceExecutor");
        jobManagerController = (JobManagerController) applicationContext.getBean("jobManagerController");
    }

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        JobWebSocketService.applicationContext = applicationContext;

    }
}

