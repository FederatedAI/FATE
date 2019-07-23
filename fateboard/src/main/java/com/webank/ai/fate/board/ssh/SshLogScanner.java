package com.webank.ai.fate.board.ssh;

import com.alibaba.fastjson.JSON;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.jcraft.jsch.Channel;
import com.webank.ai.fate.board.log.LogFileService;
import com.webank.ai.fate.board.log.LogScanner;
import com.webank.ai.fate.board.pojo.SshInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;


public class SshLogScanner implements Runnable, LogScanner {


    Logger logger = LoggerFactory.getLogger(SshLogScanner.class);

    javax.websocket.Session webSocketSession;

    boolean needStop = false;

    String jobId;

    String componentId;

    String role;

    String partyId;

    String type;

    String filePath;

    LogFileService logFileService;

    SshInfo sshInfo;

    Integer beginLine;

    Integer batchSize = 100;

    public SshLogScanner(javax.websocket.Session webSocketSession,
                         LogFileService logFileService,
                         SshInfo sshInfo,
                         String jobId, String componentId, String type, String role, String partyId, Integer beginLine) {
        Preconditions.checkArgument(jobId != null && !jobId.equals(""));
        this.jobId = jobId;
        Preconditions.checkArgument(componentId != null && !componentId.equals(""));
        this.componentId = componentId;
        this.filePath = logFileService.buildFilePath(jobId, componentId, type, role, partyId);
        this.sshInfo = sshInfo;
        Preconditions.checkArgument(type != null && !type.equals(""));
        this.type = type;
        this.webSocketSession = webSocketSession;
        this.logFileService = logFileService;
        this.beginLine = beginLine;

    }


    public void pullLog() throws IOException {
        BufferedReader reader = null;
        Channel wcChannel = null;
        Channel tailChannel = null;

        try {

            int readLine = beginLine;
            String ip = sshInfo.getIp();
            Channel channel = null;
            InputStream inputStream = null;

            while (webSocketSession.isOpen() && !needStop) {
                String cmd = logFileService.buildCommand(beginLine, filePath);
                try {
                    channel = logFileService.getRemoteLogStream(sshInfo, cmd);
                    inputStream = channel.getInputStream();
                    reader = new BufferedReader(new InputStreamReader(inputStream));
                    String content = reader.readLine();
                    List<Map> result = Lists.newArrayList();
                    int batchCount = 0;
                    while ((content = reader.readLine()) != null && !needStop) {
                        readLine++;
                        if(logger.isDebugEnabled()) {
                            logger.info("remote file readline {}", readLine);
                        }
                        Map jsonContent = LogFileService.toLogMap(content, readLine);
                        result.add(jsonContent);
                        if (result.size() >= batchSize) {
                            if (webSocketSession.isOpen()) {
                                webSocketSession.getBasicRemote().sendText(JSON.toJSONString(result));
                            }
                            result.clear();

                        }

                    }
                    if (webSocketSession.isOpen()) {
                        if (result.size() != 0) {
                            webSocketSession.getBasicRemote().sendText(JSON.toJSONString(result));
                        }
                    }

                } finally {
                    if (channel != null) {
                        channel.disconnect();
                    }
                    if (inputStream != null) {
                        inputStream.close();
                    }
                }
                if (readLine > beginLine) {
                    Thread.sleep(200);
                } else {
                    Thread.sleep(10000);
                }
                beginLine = readLine;
            }
        } catch (Throwable e) {

            logger.error("close web socket session",e);
            webSocketSession.close();
        } finally {
            if (wcChannel != null) {
                wcChannel.disconnect();
            }
            if (tailChannel != null) {
                tailChannel.disconnect();
            }
        }

    }

    @Override
    public void run() {

        try {
            this.pullLog();
        } catch (Exception e) {
            e.printStackTrace();

        } finally {
            try {
                webSocketSession.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public boolean isNeedStop() {
        return needStop;
    }

    @Override
    public void setNeedStop(boolean needStop) {
        this.needStop = needStop;

    }
}
