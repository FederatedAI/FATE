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
package com.webank.ai.fate.board.ssh;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.jcraft.jsch.Channel;
import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;
import com.webank.ai.fate.board.pojo.SshInfo;
import com.webank.ai.fate.board.utils.Dict;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


@Service
public class SshService implements InitializingBean {

    ExecutorService flushExecutor  = Executors.newSingleThreadExecutor();

    private     Properties sshInfoToProperties(){
        Properties properties = new Properties();

        sshInfoMap.values().forEach(sshInfo -> {
            StringBuilder  sb = new StringBuilder();
            String ip = sshInfo.getIp();
            String password = sshInfo.getPassword();
            Integer port = sshInfo.getPort();
            String  user = sshInfo.getUser();
            sb.append(user!=null?user:"").append("|").append(password!=null?password:"").append("|")
                    .append(port!=null?port:"");
            properties.put(ip,sb.toString());

        });

        return  properties;


    }

    public void addSShInfo(SshInfo  sshInfo){
        if(sshInfo!=null)
         sshInfoMap.put(sshInfo.getIp(),sshInfo);
    }


    public   void  flushToFile(){

        Properties  properties = sshInfoToProperties();

        flushExecutor.execute(()->{

            try {
                OutputStream  outputStream = getOutputStream();
                properties.store(outputStream, "store");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

        });

    }

    static Map<String, Session> sessionMap = Maps.newHashMap();
    Logger logger = LoggerFactory.getLogger(SshService.class);
    Map<String, SshInfo> sshInfoMap = Maps.newHashMap();
    private String pubKeyPath = "";
    public     Map<String, SshInfo>   getAllsshInfo(){
        return  sshInfoMap;
    }

    public SshInfo getSSHInfo(String ip) {
        return this.sshInfoMap.get(ip);
    }

    public void load(InputStream inputStream) throws IOException {

        Properties properties = new Properties();
        properties.load(inputStream);

        Set<String> sets = properties.stringPropertyNames();

        sets.forEach(key -> {
            try {
                String values = properties.getProperty(key);
                SshInfo sshInfo = new SshInfo();
                String[] params = values.split("\\|");
                sshInfo.setIp(key);
                if (params.length > 0) {
                    sshInfo.setUser(params[0]);
                    sshInfo.setPassword(params[1]);
                    sshInfo.setPort(new Integer(params[2]));
                }
                sshInfoMap.put(key, sshInfo);
            } catch (Exception e) {
                e.printStackTrace();
                logger.error("parse ssh info error", e);
            }
        });
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        try {
            String filePath = System.getProperty(Dict.SSH_CONFIG_FILE);
            if (filePath == null || "".equals(filePath)) {
                ClassPathResource classPathResource = new ClassPathResource("ssh.properties");
                load(classPathResource.getInputStream());
            } else {
                File file = new File(filePath + "/ssh.properties");
                Preconditions.checkArgument(file.exists() && file.isFile());
                load(new FileInputStream(file));
            }
            ;
        }catch(Exception e){
            logger.error("load ssh config file error",e);
        }
    }

    public ChannelExec executeCmd(Session session, String cmd) throws Exception {

        Preconditions.checkArgument(session != null, "ssh session is null");
        Preconditions.checkArgument(cmd != null);
        Channel channel = null;

        try {

            if (session.isConnected()) {
                channel = session.openChannel("exec");
                logger.info("prepare to execute cmd {}", cmd);
                ((ChannelExec) channel).setCommand(cmd);
                channel.setInputStream(null);
                channel.connect();
            }

        } catch (Exception e) {

            logger.error("ssh execute cmd {} error", cmd);

            e.printStackTrace();
            channel.disconnect();
            throw e;

        }

        return (ChannelExec) channel;

    }

    public Session connect(SshInfo sshInfo) throws Exception {


        return this.connect(sshInfo.getUser(), sshInfo.getPassword(), sshInfo.getIp(), new Integer(sshInfo.getPort()),5000);

    }

    public Session connect(SshInfo sshInfo,int timeout) throws Exception {

        Preconditions.checkArgument(sshInfo != null, "sshInfo is null");

        String currentUser = System.getProperty("user.name");

        return this.connect(sshInfo.getUser(), sshInfo.getPassword(), sshInfo.getIp(), new Integer(sshInfo.getPort()),timeout);

    }

    public Session connect(String user, String passwd, String host, int port,int  timeout) throws Exception {


        String sessionKey = new StringBuilder().append(user).append("_").append(host).append("_").append(port).toString();
        Session session = sessionMap.get(sessionKey);

        if (session != null && !session.isConnected()) {
            sessionMap.remove(sessionKey);
        }

        if (session != null && session.isConnected()) {
            return session;
        } else {
            JSch jsch = new JSch();


            session = jsch.getSession(user, host, port);
            if (session == null) {
                throw new Exception("session is null");
            }



            session.setPassword(passwd);
            java.util.Properties config = new java.util.Properties();
            config.put("StrictHostKeyChecking", "no");
            session.setConfig(config);
            try {
                session.connect(timeout);
            } catch (Exception e) {
                e.printStackTrace();

                logger.error("ssh connect error {} password {}", sessionKey, passwd);
                throw new Exception("ssh connect error");
            }
            sessionMap.put(sessionKey, session);
        }
        return session;
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
