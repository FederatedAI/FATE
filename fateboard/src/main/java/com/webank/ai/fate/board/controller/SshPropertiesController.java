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
import com.alibaba.fastjson.JSONObject;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.jcraft.jsch.Session;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.pojo.SshInfo;
import com.webank.ai.fate.board.ssh.SshService;
import com.webank.ai.fate.board.utils.Dict;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping(value = "/ssh")

public class SshPropertiesController {
    private final static Logger logger = LoggerFactory.getLogger(SshPropertiesController.class);

    @Autowired
    SshService sshService;

    @RequestMapping(value = "/checkStatus", method = RequestMethod.GET)
    public ResponseResult checkSShStatus() throws IOException, InterruptedException {

        Map<String ,SshInfo>  data = sshService.getAllsshInfo();

        CountDownLatch  countDownLatch  = new CountDownLatch(data.size());

        data.forEach((k,v)->{
            new  Thread(()->{
                try {
                    Session session = null;
                    try {
                        session = sshService.connect(v,1000);
                    } catch (Exception e) {
                        v.setStatus("0");
                    }
                    if (session != null) {
                        v.setStatus("1");
                    }
                }finally {
                    countDownLatch.countDown();
                }
            }).start();
        });

        countDownLatch.await(8, TimeUnit.SECONDS);


        return new ResponseResult<>(ErrorCode.SUCCESS, data);
    }
    private  Map<String, SshInfo>  getAll(){

       return   sshService.getAllsshInfo();

    }
    @RequestMapping(value = "/all", method = RequestMethod.GET)
    public ResponseResult readAll() throws Exception {

        return new ResponseResult<Map>(ErrorCode.SUCCESS, getAll());
    }

    private String checkStatus( String ip)  {
        String status = null;
        SshInfo sshInfo = sshService.getSSHInfo(ip);
        if(sshInfo==null) {
            return "0";
        }
        Session session = null;
        try {
            session = sshService.connect(sshInfo,1000);
        } catch (Exception e) {
            e.printStackTrace();
            status = "0";
        }
        if (session != null) {
            status = "1";
        }
        return status;
    }


    @RequestMapping(value = "/ssh", method = RequestMethod.GET)
    public ResponseResult readValue(@RequestBody String params) throws Exception {
        JSONObject jsonObject = JSON.parseObject(params);
        String ip = jsonObject.getString(Dict.SSH_IP);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(ip));
        HashMap<Object, List> data = Maps.newHashMap();
        List<String> sshInformation = new LinkedList<>();

        return new ResponseResult<>(ErrorCode.SUCCESS, data);
    }

    @RequestMapping(value = "/ssh", method = RequestMethod.DELETE)
    public ResponseResult removeValue(@RequestBody String params) throws IOException {
        JSONObject jsonObject = JSON.parseObject(params);
        String ip = jsonObject.getString(Dict.SSH_IP);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(ip));
        sshService.getAllsshInfo().remove(ip);
        sshService.flushToFile();
        return new ResponseResult(ErrorCode.SUCCESS);
    }

    @RequestMapping(value = "/ssh", method = RequestMethod.POST)
    public ResponseResult addProperties(@RequestBody String params) throws IOException {
        JSONObject jsonObject = JSON.parseObject(params);
        String ip = jsonObject.getString(Dict.SSH_IP);
        String user = jsonObject.getString(Dict.SSH_USER);
        String password = jsonObject.getString(Dict.SSH_PASSWORD);
        String port = jsonObject.getString(Dict.SSH_PORT);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(ip, user, password, port));
        SshInfo  sshInfo = new SshInfo();
        sshInfo.setPort(new Integer(port));
        sshInfo.setIp(ip);
        sshInfo.setUser(user);
        sshInfo.setPassword(password);
        sshService.addSShInfo(sshInfo);
        sshService.flushToFile();
        return new ResponseResult<Map>(ErrorCode.SUCCESS, sshService.getAllsshInfo());

    }

}
