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

import com.jcraft.jsch.*;
import com.jcraft.jsch.ChannelSftp.LsEntry;
import com.webank.ai.fate.board.pojo.SshInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;


public class SftpUtils {
    private static Logger log = LoggerFactory.getLogger(SftpUtils.class.getName());

    public static ChannelSftp connect(SshInfo sshInfo) {
        try {
            JSch jsch = new JSch();
            Session sshSession = jsch.getSession(sshInfo.getUser(), sshInfo.getIp(), sshInfo.getPort());
            if (log.isInfoEnabled()) {
                log.info("Session created.");
            }
            sshSession.setPassword(sshInfo.getPassword());
            Properties sshConfig = new Properties();
            sshConfig.put("StrictHostKeyChecking", "no");
            sshSession.setConfig(sshConfig);
            sshSession.connect();
            if (log.isInfoEnabled()) {
                log.info("Session connected.");
            }
            Channel channel = sshSession.openChannel("sftp");
            channel.connect();
            if (log.isInfoEnabled()) {
                log.info("Opening Channel.");
            }
            return (ChannelSftp) channel;

        } catch (Exception e) {
            log.error("sftp connect error",e);
        }

        return null;
    }


    public static void disconnect(ChannelSftp channelSftp) {
        if (channelSftp != null) {
            if (channelSftp.isConnected()) {
                channelSftp.disconnect();
                if (log.isInfoEnabled()) {
                    log.info("sftp is closed already");
                }
            }

            try {
                if (channelSftp.getSession() != null) {
                    if (channelSftp.getSession().isConnected()) {
                        channelSftp.getSession().disconnect();
                    }
                }
            } catch (JSchException e) {
                e.printStackTrace();
            }
        }

    }


    public static List<String> batchDownLoadFile(SshInfo sshInfo,
                                                 String remotePath,
                                                 String localPath,
                                                 String fileFormat,
                                                 String fileEndFormat,
                                                 boolean del) {


        ChannelSftp channelSftp = null;
        channelSftp = connect(sshInfo);
        try {
            return batchDownLoadFileInner(channelSftp, remotePath, localPath, fileFormat, fileEndFormat, del);
        } finally {
            if (channelSftp != null) {
                try {
                    if (channelSftp.getSession() != null) {
                        channelSftp.getSession().disconnect();
                    }
                } catch (JSchException e) {
                    e.printStackTrace();
                }
                channelSftp.disconnect();
            }
        }
    }


    private static List<String> batchDownLoadFileInner(
            ChannelSftp channelSftp,
            String remotePath,
            String localPath,
            String fileFormat,
            String fileEndFormat,
            boolean del) {
        mkdirs(localPath);


        List<String> filenames = new ArrayList<String>();

        try {

            Vector v = listFiles(channelSftp, remotePath);
            if (v.size() > 0) {
                Iterator it = v.iterator();
                while (it.hasNext()) {
                    LsEntry entry = (LsEntry) it.next();
                    String filename = entry.getFilename();
                    SftpATTRS attrs = entry.getAttrs();
                    if (!attrs.isDir()) {
                        boolean flag = false;
                        String localFileName = localPath + filename;
                        fileFormat = fileFormat == null ? "" : fileFormat
                                .trim();
                        fileEndFormat = fileEndFormat == null ? ""
                                : fileEndFormat.trim();
                        if (fileFormat.length() > 0 && fileEndFormat.length() > 0) {
                            if (filename.startsWith(fileFormat) && filename.endsWith(fileEndFormat)) {
                                flag = downloadFile(channelSftp, remotePath, filename, localPath, filename);
                                if (flag) {
                                    filenames.add(localFileName);
                                    if (flag && del) {
                                        deleteSFTP(channelSftp, remotePath, filename);
                                    }
                                }
                            }
                        } else if (fileFormat.length() > 0 && "".equals(fileEndFormat)) {
                            if (filename.startsWith(fileFormat)) {
                                flag = downloadFile(channelSftp, remotePath, filename, localPath, filename);
                                if (flag) {
                                    filenames.add(localFileName);
                                    if (flag && del) {
                                        deleteSFTP(channelSftp, remotePath, filename);
                                    }
                                }
                            }
                        } else if (fileEndFormat.length() > 0 && "".equals(fileFormat)) {
                            if (filename.endsWith(fileEndFormat)) {
                                flag = downloadFile(channelSftp, remotePath, filename, localPath, filename);
                                if (flag) {
                                    filenames.add(localFileName);
                                    if (flag && del) {
                                        deleteSFTP(channelSftp, remotePath, filename);
                                    }
                                }
                            }
                        } else {
                            flag = downloadFile(channelSftp, remotePath, filename, localPath, filename);
                            if (flag) {
                                filenames.add(localFileName);
                                if (flag && del) {
                                    deleteSFTP(channelSftp, remotePath, filename);
                                }
                            }
                        }
                    } else {
                        if (!filename.equals(".") && !filename.equals("..")) {
                            batchDownLoadFileInner(channelSftp,
                                    remotePath + filename + "/",
                                    localPath + filename + "/",
                                    fileFormat,
                                    fileEndFormat,
                                    del);

                        }
                    }
                }
            }
            if (log.isInfoEnabled()) {
                log.info("download file is success:remotePath=" + remotePath
                        + "and localPath=" + localPath + ",file size is"
                        + v.size());
            }
        } catch (SftpException e) {
            e.printStackTrace();
        }

        return filenames;
    }


    public static boolean downloadFile(ChannelSftp sftp, String remotePath, String remoteFileName, String localPath, String localFileName) {
        FileOutputStream fieloutput = null;
        try {
            if (log.isInfoEnabled()) {
                log.info("remote file {} : localfile {}", remotePath + remoteFileName, localPath + localFileName);
            }
            File file = new File(localPath + localFileName);
            fieloutput = new FileOutputStream(file);
            sftp.get(remotePath + remoteFileName, fieloutput);
            return true;
        } catch (FileNotFoundException e) {
            log.error("file not find",e);
        } catch (SftpException e) {
            log.error("sftp error",e);
        } finally {
            if (null != fieloutput) {
                try {
                    fieloutput.close();
                } catch (IOException e) {
                    log.error("sftp error",e);
                }
            }
        }
        return false;
    }


    public static boolean deleteFile(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            return false;
        }
        if (!file.isFile()) {
            return false;
        }
        boolean rs = file.delete();
        if (rs && log.isInfoEnabled()) {
            log.info("delete file success from local.");
        }
        return rs;
    }

    public static boolean isDirExist(ChannelSftp channelSftp, String directory) {
        boolean isDirExistFlag = false;
        try {
            SftpATTRS sftpATTRS = channelSftp.lstat(directory);
            isDirExistFlag = true;
            return sftpATTRS.isDir();
        } catch (Exception e) {
            if (e.getMessage().toLowerCase().equals("no such file")) {
                isDirExistFlag = false;
            }
        }
        return isDirExistFlag;
    }


    public static void deleteSFTP(ChannelSftp channelSftp, String directory, String deleteFile) {
        try {
            // sftp.cd(directory);
            channelSftp.rm(directory + deleteFile);
            if (log.isInfoEnabled()) {
                log.info("delete file success from sftp.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void mkdirs(String path) {
        File f = new File(path);

        String fs = f.getParent();

        f = new File(fs);

        if (!f.exists()) {
            f.mkdirs();
        }
    }

    public static Vector listFiles(ChannelSftp sftp, String directory) throws SftpException {
        return sftp.ls(directory);
    }

}