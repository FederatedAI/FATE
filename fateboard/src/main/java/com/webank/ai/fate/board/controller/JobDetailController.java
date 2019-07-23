package com.webank.ai.fate.board.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.google.common.base.Preconditions;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
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


@Controller
@RequestMapping(value = "/v1")
public class JobDetailController {

    private final Logger logger = LoggerFactory.getLogger(JobDetailController.class);

    @Autowired
    HttpClientPool httpClientPool;
    @Autowired
    TaskManagerService taskManagerService;

    @Value("${fate.url}")
    String fateUrl;


    /**
     * get running parameters
     *
     * @param param
     * @return
     */
    @ResponseBody
    @RequestMapping(value = "/tracking/component/metrics", method = RequestMethod.POST)
    public ResponseResult getMetaInfo(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_METRIC, jsonObject.toJSONString());
        return ResponseUtil.buildResponse(result, Dict.DATA);
    }

    ;

    /**
     * get index parameters
     *
     * @param param
     * @return
     */
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
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName, metricName, metricNamespace));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_METRIC_DATA, jsonObject.toJSONString());
        return ResponseUtil.buildResponse(result, null);
    }

    /**
     * get graph/table parameters
     *
     * @param param
     * @return
     */
    @RequestMapping(value = "/tracking/component/parameters", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getDetailInfo(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        String componentName = jsonObject.getString(Dict.COMPONENT_NAME);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_COPONENT_PARAMETERS, jsonObject.toJSONString());
        Preconditions.checkArgument(result != null);
        return ResponseUtil.buildResponse(result, Dict.DATA);

    }

    /**
     * get dag dependencies
     *
     * @param param
     * @return
     */
    @RequestMapping(value = "/pipeline/dag/dependencies", method = RequestMethod.POST)
    @ResponseBody
    public ResponseResult getDagDependencies(@RequestBody String param) {
        JSONObject jsonObject = JSON.parseObject(param);
        String jobId = jsonObject.getString(Dict.JOBID);
        String role = jsonObject.getString(Dict.ROLE);
        String partyId = jsonObject.getString(Dict.PARTY_ID);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_DAG_DEPENDENCY, jsonObject);
        Preconditions.checkArgument(result != null);
        JSONObject resultObject = JSON.parseObject(result);
        Integer retcode = resultObject.getInteger(Dict.RETCODE);

        if (retcode == 0) {
            JSONObject data = resultObject.getJSONObject(Dict.DATA);
            JSONArray components_list = data.getJSONArray(Dict.COMPONENT_LIST);
            ArrayList<Map> componentList = new ArrayList<>();
            for (Object o : components_list) {
                HashMap<String, String> component = new HashMap<>();
                component.put(Dict.COMPONENT_NAME, (String) o);
                String taskStatus = taskManagerService.findTaskStatus(jobId, (String) o);
                component.put(Dict.STATUS, taskStatus);
                componentList.add(component);
            }
            data.put(Dict.COMPONENT_LIST, componentList);
            return new ResponseResult<>(ErrorCode.SUCCESS, data);

        } else {
            return new ResponseResult<>(retcode, resultObject);
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
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_OUTPUT_MODEL, jsonObject.toJSONString());
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
        Preconditions.checkArgument(StringUtils.isNoneEmpty(jobId, role, partyId, componentName));
        jsonObject.put(Dict.PARTY_ID, new Integer(partyId));
        String result = httpClientPool.post(fateUrl + Dict.URL_OUTPUT_DATA, jsonObject.toJSONString());
        return ResponseUtil.buildResponse(result, null);
    }
}
