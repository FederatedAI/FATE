
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
package com.webank.ai.fate.board.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.pojo.Job;
import com.webank.ai.fate.board.pojo.JobWithBLOBs;
import com.webank.ai.fate.board.services.JobManagerService;
import com.webank.ai.fate.board.utils.*;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

@CrossOrigin
@RestController
@RequestMapping(value = "/job")
public class JobManagerController {
    private final Logger logger = LoggerFactory.getLogger(JobManagerController.class);

    @Autowired
    JobManagerService jobManagerService;

    @Autowired
    HttpClientPool httpClientPool;

    @Value("${fateflow.url}")
    String fateUrl;
    @Autowired
    ThreadPoolTaskExecutor asyncServiceExecutor;

    @RequestMapping(value = "/query/status", method = RequestMethod.GET)
    public ResponseResult queryJobStatus() {
        List<Job> jobs = jobManagerService.queryJobStatus();
        return new ResponseResult<>(ErrorCode.SUCCESS, jobs);
    }

    @RequestMapping(value = "/v1/pipeline/job/stop", method = RequestMethod.POST)
    public ResponseResult stopJob(@RequestBody String param) {

        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_JOB_STOP, jsonObject.toJSONString());
        return ResponseUtil.buildResponse(result, null);

    }

    @RequestMapping(value = "/tracking/job/data_view", method = RequestMethod.POST)
    public ResponseResult queryJobDataset(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_JOB_DATAVIEW, jsonObject.toJSONString());
        return ResponseUtil.buildResponse(result, Dict.DATA);
    }


    @RequestMapping(value = "/query/{jobId}/{role}/{partyId}", method = RequestMethod.GET)
    public ResponseResult queryJobById(@PathVariable("jobId") String jobId,
                                       @PathVariable("role") String role,
                                       @PathVariable("partyId") String partyId
    ) {
        HashMap<String, Object> resultMap = new HashMap<>();
        JobWithBLOBs jobWithBLOBs = jobManagerService.queryJobByConditions(jobId, role, partyId);
        if (jobWithBLOBs == null) {
            return new ResponseResult<>(ErrorCode.INCOMING_PARAM_ERROR);

        }
        Map params = Maps.newHashMap();
        params.put(Dict.JOBID, jobId);
        params.put(Dict.ROLE, role);
        params.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_JOB_DATAVIEW, JSON.toJSONString(params));
        JSONObject data = JSON.parseObject(result).getJSONObject(Dict.DATA);
        resultMap.put(Dict.JOB, jobWithBLOBs);
        resultMap.put(Dict.DATASET, data);
        return new ResponseResult<>(ErrorCode.SUCCESS, resultMap);
    }


    @RequestMapping(value = "/query/totalrecord", method = RequestMethod.GET)
    public ResponseResult queryTotalRecord() {
        long count = jobManagerService.count();
        return new ResponseResult<>(ErrorCode.SUCCESS, count);
    }


    @RequestMapping(value = "/query/page", method = RequestMethod.POST)
    public ResponseResult queryJobByPage(@RequestBody String pageParams) {

        JSONObject pageObject = JSON.parseObject(pageParams);
        String jobId = pageObject.getString(Dict.JOBID);
        String partyId = pageObject.getString(Dict.PARTY_ID);

        JSONArray roleJsonArray = pageObject.getJSONArray(Dict.ROLE);
        List<String> roles = roleJsonArray.toJavaList(String.class);
        JSONArray jobJsonArray = pageObject.getJSONArray(Dict.JOB_STATUS);
        List<String> jobStatus = jobJsonArray.toJavaList(String.class);

        String startTime = pageObject.getString(Dict.START_TIME);
        String endTime = pageObject.getString(Dict.END_TIME);
        Long pageNum = pageObject.getLong(Dict.PAGENUM);
        Long pageSize = pageObject.getLong(Dict.PAGESIZE);

        if (pageNum == null || pageSize == null) {
            throw new IllegalArgumentException();
        }


        long totalRecord = jobManagerService.totalCount(jobId, roles, partyId, jobStatus);
        PageBean<Map> listPageBean = new PageBean<>(pageNum, pageSize, totalRecord);
        long startIndex = listPageBean.getStartIndex();
        List<JobWithBLOBs> jobWithBLOBs = jobManagerService.queryPageByCondition(jobId, roles, partyId, jobStatus, startTime, endTime, startIndex, pageSize);
        ArrayList<Map> jobList = new ArrayList<>();
        Map<JobWithBLOBs, Future> jobDataMap = new LinkedHashMap<>();

        for (JobWithBLOBs jobWithBLOB : jobWithBLOBs) {
            ListenableFuture<?> future = ThreadPoolTaskExecutorUtil.submitListenable(this.asyncServiceExecutor, (Callable<JSONObject>) () -> {
                String jobId1 = jobWithBLOB.getfJobId();
                String role1 = jobWithBLOB.getfRole();
                String partyId1 = jobWithBLOB.getfPartyId();
                HashMap<String, String> jobParams = Maps.newHashMap();
                jobParams.put(Dict.JOBID, jobId1);
                jobParams.put((Dict.ROLE), role1);
                jobParams.put(Dict.PARTY_ID, partyId1);
                String result = httpClientPool.post(fateUrl + Dict.URL_JOB_DATAVIEW, JSON.toJSONString(jobParams));
                JSONObject data = JSON.parseObject(result).getJSONObject(Dict.DATA);


                return data;
            }, new int[]{500, 1000}, new int[]{3, 3});
            jobDataMap.put(jobWithBLOB, future);
        }
        jobDataMap.forEach((k, v) -> {
            HashMap<String, Object> stringObjectHashMap = new HashMap<>();
            stringObjectHashMap.put(Dict.JOB, k);
            try {
                stringObjectHashMap.put(Dict.DATASET, v.get());
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            jobList.add(stringObjectHashMap);
        });
        listPageBean.setList(jobList);
        return new ResponseResult<>(ErrorCode.SUCCESS, listPageBean);
    }


}
