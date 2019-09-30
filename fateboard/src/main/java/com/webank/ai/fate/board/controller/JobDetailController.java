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
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.pojo.Task;
import com.webank.ai.fate.board.services.TaskManagerService;
import com.webank.ai.fate.board.utils.Dict;
import com.webank.ai.fate.board.utils.HttpClientPool;
import com.webank.ai.fate.board.utils.ResponseUtil;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static com.webank.ai.fate.board.global.ErrorCode.FATEFLOW_ERROR_CONNECTION;
import static com.webank.ai.fate.board.global.ErrorCode.REQUEST_PARAMETER_ERROR;


@Controller
@RequestMapping(value = "/v1")
public class JobDetailController {

    private final Logger logger = LoggerFactory.getLogger(JobDetailController.class);

    @Autowired
    HttpClientPool httpClientPool;
    @Autowired
    TaskManagerService taskManagerService;

    @Value("${fateflow.url}")
    String fateUrl;

    @ResponseBody
    @RequestMapping(value = "/tracking/component/metrics", method = RequestMethod.POST)
    public ResponseResult getMetaInfo(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }

        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));

        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_METRIC, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }
        return ResponseUtil.buildResponse(result, Dict.DATA);
    }

    @RequestMapping(value = "/tracking/component/metric_data", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getMetricInfo(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        String metricNamespace = jsonObject.getString(Dict.METRIC_NAMESPACE);
        String metricName = jsonObject.getString(Dict.METRIC_NAME);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName, metricName, metricNamespace));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_METRIC_DATA, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }
        return ResponseUtil.buildResponse(result, null);
    }


    @RequestMapping(value = "/tracking/component/parameters", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getDetailInfo(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_PARAMETERS, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }

        return ResponseUtil.buildResponse(result, Dict.DATA);

    }


    @RequestMapping(value = "/pipeline/dag/dependencies", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getDagDependencies(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }

        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_DAG_DEPENDENCY, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }
        if ((result == null) || (result == "")) {
            return new ResponseResult<>(ErrorCode.FATEFLOW_ERROR_NULL_RESULT);
        }

        JSONObject resultObject = JSON.parseObject(result);
        Integer retcode = resultObject.getInteger(Dict.RETCODE);
        if (retcode == null) {
            return new ResponseResult<>(ErrorCode.FATEFLOW_ERROR_WRONG_RESULT);
        }

        if (retcode == 0) {
            JSONObject data = resultObject.getJSONObject(Dict.DATA);
            JSONArray components_list = data.getJSONArray(Dict.COMPONENT_LIST);
            ArrayList<Map> componentList = new ArrayList<>();

            for (Object o : components_list) {
                HashMap<String, Object> component = new HashMap<>();
                component.put(Dict.COMPONENT_NAME, (String) o);
                Task task = taskManagerService.findTask(jobId, role, (String) o);
                String taskStatus = null;
                Long createTime = null;
                if (task != null) {
                    taskStatus = task.getfStatus();
                    createTime = task.getfCreateTime();
                }

                component.put(Dict.STATUS, taskStatus);
                component.put(Dict.TIME, createTime);
                componentList.add(component);
            }

            data.put(Dict.COMPONENT_LIST, componentList);
            return new ResponseResult<>(ErrorCode.SUCCESS, data);

        } else {
            return new ResponseResult<>(retcode, resultObject.getString(Dict.RETMSG));
        }
    }

    @RequestMapping(value = "/tracking/component/output/model", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getModel(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_OUTPUT_MODEL, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }
        return ResponseUtil.buildResponse(result, null);
    }

    @RequestMapping(value = "/tracking/component/output/data", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getData(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        try {
            Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(REQUEST_PARAMETER_ERROR);
        }
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = null;
        try {
            result = httpClientPool.post(fateUrl + Dict.URL_OUTPUT_DATA, jsonObject.toJSONString());
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(FATEFLOW_ERROR_CONNECTION);
        }
        return ResponseUtil.buildResponse(result, null);
    }
}
