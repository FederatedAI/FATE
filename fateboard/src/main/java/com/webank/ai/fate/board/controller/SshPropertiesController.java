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

        HashMap<Object, SshInfo> data = Maps.newHashMap();

        Properties properties = new Properties();
        try( InputStream inputStream = this.getInputStream()) {
            properties.load(inputStream);
            Enumeration<?> enumeration = properties.propertyNames();
            int size = properties.size();

            while (enumeration.hasMoreElements()) {

                try {
                    SshInfo sshInfo = new SshInfo();
                    String status = "0";
                    String ip = (String) enumeration.nextElement();
                    status = checkStatus(ip);
                    String sshValue = properties.getProperty(ip);
                    String[] params = sshValue.split("\\|");
                    sshInfo.setIp(ip);
                    if (params.length > 0) {
                        sshInfo.setUser(params[0]);
                        sshInfo.setPassword(params[1]);
                        sshInfo.setPort(new Integer(params[2]));
                        sshInfo.setStatus(status);
                    }
                    data.put(ip, sshInfo);
                }catch(Exception e){

                }
            }
        }
        int  size =  data.size();

        CountDownLatch  countDownLatch  = new CountDownLatch(size);
        data.forEach((k,v)->{
            new  Thread(()->{

                try {
                    Session session = null;
                    try {
                        session = sshService.connect(v);
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



        @RequestMapping(value = "/all", method = RequestMethod.GET)
    public ResponseResult readAll() throws Exception {


        HashMap<Object, SshInfo> data = Maps.newHashMap();

        Properties properties = new Properties();
       try( InputStream inputStream = this.getInputStream()) {
           properties.load(inputStream);
           Enumeration<?> enumeration = properties.propertyNames();
           int size = properties.size();

           while (enumeration.hasMoreElements()) {
               SshInfo  sshInfo  =   new SshInfo();


               String ip = (String) enumeration.nextElement();

               String sshValue = properties.getProperty(ip);

               String[] params = sshValue.split("\\|");
               sshInfo.setIp(ip);
               if (params.length > 0) {
                   sshInfo.setUser(params[0]);
                   sshInfo.setPassword(params[1]);
                   sshInfo.setPort(new Integer(params[2]));
               }
               data.put(ip, sshInfo);
           }

       }


        return new ResponseResult<Map>(ErrorCode.SUCCESS, data);
    }

    private String checkStatus( String ip)  {
        String status = null;
        SshInfo sshInfo = sshService.getSSHInfo(ip);
        if(sshInfo==null) {
            return "0";
        }
        Session session = null;
        try {
            session = sshService.connect(sshInfo);
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

        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        String value = properties.getProperty(ip);
        String status = checkStatus( ip);

        sshInformation.add(value);
        sshInformation.add(status);
        data.put(ip, sshInformation);

        inputStream.close();
        return new ResponseResult<>(ErrorCode.SUCCESS, data);
    }

    @RequestMapping(value = "/ssh", method = RequestMethod.DELETE)
    public ResponseResult removeValue(@RequestBody String params) throws IOException {
        JSONObject jsonObject = JSON.parseObject(params);
        String ip = jsonObject.getString(Dict.SSH_IP);
        Preconditions.checkArgument(StringUtils.isNoneEmpty(ip));

        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        properties.remove(ip);


        OutputStream writer = this.getOutputStream();
        properties.store(writer, "delete '" + ip + "' value");

        sshService.load(inputStream);
        writer.close();
        inputStream.close();

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

        HashMap<Object, List> data = Maps.newHashMap();
        List<String> sshInformation = new LinkedList<>();


        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        String connectInformation = user + "|" + password + "|" + port;
        properties.setProperty(ip, connectInformation);
        OutputStream writer = this.getOutputStream();
        properties.store(writer, "add" + "  key:" + ip + ", value" + connectInformation);

        String status = checkStatus( ip);
        sshInformation.add(connectInformation);
        sshInformation.add(status);
        data.put(ip, sshInformation);

        sshService.load(inputStream);
        writer.close();
        inputStream.close();

        return new ResponseResult<Map>(ErrorCode.SUCCESS, data);

    }


    private InputStream getInputStream() throws IOException {
        String filePath = System.getProperty(Dict.SSH_CONFIG_FILE);
        if (filePath == null || "".equals(filePath)) {
            ClassPathResource classPathResource = new ClassPathResource("ssh.properties");
            return classPathResource.getInputStream();
        } else {
            File file = new File(filePath + "/ssh.properties");
            Preconditions.checkArgument(file.exists() && file.isFile());
            return new BufferedInputStream(new FileInputStream(file));
        }

    }


    private OutputStream getOutputStream() throws FileNotFoundException {
        String filePath = System.getProperty(Dict.SSH_CONFIG_FILE);
        if (filePath == null || "".equals(filePath)) {
            ClassPathResource classPathResource = new ClassPathResource("ssh.properties");
            String path = classPathResource.getPath();
            return new BufferedOutputStream(new FileOutputStream(path));
        } else {
            File file = new File(filePath + "/ssh.properties");
            Preconditions.checkArgument(file.exists() && file.isFile());
            return new BufferedOutputStream(new FileOutputStream(file));
        }

    }

}
