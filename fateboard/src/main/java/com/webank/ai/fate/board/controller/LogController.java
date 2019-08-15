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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.log.LogFileService;
import com.webank.ai.fate.board.pojo.SshInfo;
import com.webank.ai.fate.board.ssh.SshService;
import com.webank.ai.fate.board.utils.GetSystemInfo;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.*;
import java.util.List;
import java.util.Map;


@Controller
public class LogController {
    private final Logger logger = LoggerFactory.getLogger(LogController.class);
    @Autowired
    LogFileService logFileService;
    @Autowired
    SshService sshService;

    @RequestMapping(value = "/queryLogWithSizeSSH/{jobId}/{role}/{partyId}/{componentId}/{type}/{begin}/{end}", method = RequestMethod.GET)
    @ResponseBody
    public ResponseResult queryLogWithSizeSSH(@PathVariable String componentId,
                                              @PathVariable String jobId,
                                              @PathVariable Integer begin,
                                              @PathVariable String role,
                                              @PathVariable String partyId,
                                              @PathVariable String type,
                                              @PathVariable Integer end) throws Exception {
        logger.info("parameters for " + "componentId:" + componentId + ", jobId:" + jobId + ", begin;" + begin + ", end:" + end + "type");

        String filePath = logFileService.buildFilePath(jobId, componentId, type, role, partyId);

        Preconditions.checkArgument(filePath != null && !filePath.equals(""));

        String ip = logFileService.getJobTaskInfo(jobId, componentId, role, partyId).ip;

        Preconditions.checkArgument(ip != null && !ip.equals(""));

        List<Map> logs = logFileService.getRemoteLogWithFixSize(jobId, componentId, type, role, partyId, begin, end - begin + 1);

        ResponseResult result = new ResponseResult();

        result.setData(logs);

        return result;

    }


    @RequestMapping(value = "/queryLogSize/{jobId}/{role}/{partyId}/{componentId}/{type}", method = RequestMethod.GET)
    @ResponseBody
    public ResponseResult queryLogSize(@PathVariable String componentId,
                                       @PathVariable String jobId,
                                       @PathVariable String type,
                                       @PathVariable String role,
                                       @PathVariable String partyId
    ) throws Exception {

        ResponseResult responseResult = new ResponseResult();
        responseResult.setData(0);
        String filePath = logFileService.buildFilePath(jobId, componentId, type, role, partyId);
        Preconditions.checkArgument(StringUtils.isNotEmpty(filePath));
        if (LogFileService.checkFileIsExist(filePath)) {
            Integer count = LogFileService.getLocalFileLineCount(new File(filePath));
            responseResult.setData(count);
        } else {
            String ip = logFileService.getJobTaskInfo(jobId, componentId, role, partyId).ip;
            String localIp = GetSystemInfo.getLocalIp();
            if (logger.isDebugEnabled()) {
                logger.debug("local ip {} remote ip {}", localIp, ip);
            }
            if (localIp.equals(ip) || "0.0.0.0".equals(ip) || "127.0.0.1".equals(ip)) {
                return responseResult;
            }
            logFileService.checkSshInfo(ip);
            SshInfo sshInfo = this.sshService.getSSHInfo(ip);
            try {
                Integer count = logFileService.getRemoteFileLineCount(sshInfo, filePath);
                responseResult.setData(count);
            } catch (Exception e) {
                responseResult.setData(0);
            }
        }
        return responseResult;
    }

    public long getLineNumber(File file) {
        if (file.exists()) {
            try {
                FileReader fileReader = new FileReader(file);
                LineNumberReader lineNumberReader = new LineNumberReader(fileReader);
                lineNumberReader.skip(Long.MAX_VALUE);
                long lines = lineNumberReader.getLineNumber() + 1;
                fileReader.close();
                lineNumberReader.close();
                return lines;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return 0;
    }

    List<Map> queryLog(String componentId, String jobId, String type, String role, String partyId,
                       Integer begin,
                       Integer end) throws Exception {
        String filePath = logFileService.buildFilePath(jobId, componentId, type, role, partyId);
        Preconditions.checkArgument(filePath != null && !filePath.equals(""));
        if (LogFileService.checkFileIsExist(filePath)) {
            RandomAccessFile file = null;
            List<Map> result = Lists.newArrayList();
            if (begin > end || begin <= 0) {
                throw new Exception();
            }
            String[] cmd = {"sh", "-c", "tail -n +" + begin + " " + filePath + " | head -n " + (end - begin)};
            Process process = Runtime.getRuntime().exec(cmd);
            InputStream inputStream = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            try {
                String content = null;
                int index = 0;
                do {
                    content = reader.readLine();
                    if (content != null) {
                        result.add(LogFileService.toLogMap(content, begin + index));
                    }
                    index++;
                } while (content != null);
                if (logger.isDebugEnabled()) {
                    logger.debug("execute  cmd {} return count {}", cmd, index);
                }
            } finally {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (process != null) {
                    process.destroyForcibly();
                }
            }
            return result;
        } else {
            String ip = logFileService.getJobTaskInfo(jobId, componentId, role, partyId).ip;
            logFileService.checkSshInfo(ip);
            if (StringUtils.isEmpty(ip)) {
                return null;
            }
            List<Map> logs = logFileService.getRemoteLogWithFixSize(jobId, componentId, type, role, partyId, begin, end - begin + 1);
            return logs;

        }

    }


    @RequestMapping(value = "/queryLogWithSize/{jobId}/{role}/{partyId}/{componentId}/{type}/{begin}/{end}", method = RequestMethod.GET)
    @ResponseBody
    public ResponseResult queryLogWithSize(@PathVariable String componentId,
                                           @PathVariable String jobId,
                                           @PathVariable String type,
                                           @PathVariable String role,
                                           @PathVariable String partyId,
                                           @PathVariable Integer begin,
                                           @PathVariable Integer end) throws Exception {

        logger.info("parameters for " + "componentId:" + componentId + ", jobId:" + jobId + ", begin;" + begin + ", end:" + end);

        List<Map> result = this.queryLog(componentId, jobId, type, role, partyId, begin, end);

        return new ResponseResult<>(ErrorCode.SUCCESS, result);
    }

}
