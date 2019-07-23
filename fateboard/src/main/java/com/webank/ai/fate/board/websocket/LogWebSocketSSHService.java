package com.webank.ai.fate.board.websocket;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.webank.ai.fate.board.conf.Configurator;
import com.webank.ai.fate.board.disruptor.LogFileTransferEventProducer;
import com.webank.ai.fate.board.log.LogFileService;
import com.webank.ai.fate.board.log.LogScanner;
import com.webank.ai.fate.board.log.RandomFileScanner;
import com.webank.ai.fate.board.pojo.SshInfo;
import com.webank.ai.fate.board.services.JobManagerService;
import com.webank.ai.fate.board.ssh.SshLogScanner;
import com.webank.ai.fate.board.ssh.SshService;
import com.webank.ai.fate.board.utils.GetSystemInfo;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;

import javax.websocket.*;
import javax.websocket.server.PathParam;
import javax.websocket.server.ServerEndpoint;
import java.io.File;
import java.util.Map;


@ServerEndpoint(value = "/log/{jobId}/{role}/{partyId}/{componentId}/{type}", configurator = Configurator.class)
@Component
public class LogWebSocketSSHService implements InitializingBean, ApplicationContextAware {

    private static final Logger logger = LoggerFactory.getLogger(LogWebSocketSSHService.class);
    static ApplicationContext applicationContext;

    static SshService sshService;

    static LogFileService logFileService;
    static LogFileTransferEventProducer logFileTransferEventProducer;
    static ThreadPoolTaskExecutor asyncServiceExecutor;
    final String DEFAULT_COMPONENT_ID = "default";
    Map<Session, LogScanner> sessionMap = Maps.newHashMap();
    private Integer tailNum = 1000;

    /**
     * call method when connection build
     */
    @OnOpen
    public synchronized void onOpen(Session session, @PathParam("jobId") String jobId,
                                    @PathParam("role") String role,
                                    @PathParam("partyId") String partyId,
                                    @PathParam("componentId") String componentId,
                                    @PathParam("type") String type

    ) throws Exception {


        String filePath = logFileService.buildFilePath(jobId, componentId, type, role, partyId);

        Preconditions.checkArgument(StringUtils.isNotEmpty(filePath), "file path is null");

        if (LogFileService.checkFileIsExist(filePath)) {
            File file = new File(filePath);

            long lines = LogFileService.getLocalFileLineCount(file);

            logger.info("local file {} has {} line", filePath, lines);

            long skipLines = 0;

            if (lines > tailNum) {
                skipLines = lines - tailNum;
            }
            RandomFileScanner randomFileScanner = new RandomFileScanner(file, session, skipLines);

            sessionMap.put(session, randomFileScanner);

            Thread thread = new Thread(randomFileScanner);

            thread.start();

            return;

        } else {

            logger.info("local file path {} is not exist,try to find remote file", filePath);

            LogFileService.JobTaskInfo jobTaskInfo = logFileService.getJobTaskInfo(jobId, componentId, role, partyId);

            Preconditions.checkArgument(StringUtils.isNotEmpty(jobTaskInfo.ip));

            String localIp = GetSystemInfo.getLocalIp();

            if (localIp.equals(jobTaskInfo.ip) || "0.0.0.0".equals(jobTaskInfo.ip)) {
                logger.error("local ip :{} job ip:{} log file {} not exist", localIp, jobTaskInfo.ip, filePath);

                return;
            }

            SshInfo sshInfo = sshService.getSSHInfo(jobTaskInfo.ip);

            Preconditions.checkArgument(sshInfo != null);

            if (JobManagerService.jobFinishStatus.contains(jobTaskInfo.jobStatus)) {
                String jobDir = logFileService.getJobDir(jobId);
                logFileTransferEventProducer.onData(sshInfo, jobDir, jobDir);
            }

            Integer lines = logFileService.getRemoteFileLineCount(sshInfo, filePath);

            Integer beginLine = tailNum > lines ? 0 : lines - tailNum;

            SshLogScanner sshLogScanner = new SshLogScanner(session, logFileService, sshInfo, jobId, componentId, type, role, partyId, beginLine);

            sessionMap.put(session, sshLogScanner);

            Thread thread = new Thread((Runnable) sshLogScanner);

            thread.start();

        }

    }

    /**
     * call method when connection close
     */
    @OnClose
    public void onClose(Session session) {

        logger.info("websocket session {} closed", session);

        LogScanner logScanner = sessionMap.get(session);

        if (logScanner != null) {
            logScanner.setNeedStop(true);
        }

        sessionMap.remove(session);
    }


    /**
     * call method when receive message from client
     *
     * @param message message from client
     */
    @OnMessage
    public void onMessage(String message, Session session) {

    }

    /**
     * call method when error occurs
     *
     * @param session
     * @param error
     */
    @OnError
    public void onError(Session session, Throwable error) {
        logger.error("log web socket error", error);
        error.printStackTrace();
    }

    @Override
    public void afterPropertiesSet() throws Exception {

        LogWebSocketSSHService.logFileService = (LogFileService) applicationContext.getBean("logFileService");
        LogWebSocketSSHService.sshService = (SshService) applicationContext.getBean("sshService");
        //    LogWebSocketSSHService.applicationEventPublisher  =  (ApplicationEventPublisher)applicationContext.getBean(ApplicationEventPublisher.class);
        LogWebSocketSSHService.logFileTransferEventProducer = (LogFileTransferEventProducer) applicationContext.getBean("logFileTransferEventProducer");
        asyncServiceExecutor = (ThreadPoolTaskExecutor) applicationContext.getBean("asyncServiceExecutor");

    }

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        LogWebSocketSSHService.applicationContext = applicationContext;
    }


}
