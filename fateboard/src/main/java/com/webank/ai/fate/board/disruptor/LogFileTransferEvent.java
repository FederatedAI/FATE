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
package com.webank.ai.fate.board.disruptor;

import com.webank.ai.fate.board.pojo.SshInfo;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

public class LogFileTransferEvent {

    SshInfo sshInfo;
    String sourceFilePath;
    String desFilePath;
    int status = 0;

    public LogFileTransferEvent() {

    }

    public LogFileTransferEvent(
            SshInfo sshInfo,
            String sourceFilePath,
            String desFilePath) {
        this.sshInfo = sshInfo;
        this.sourceFilePath = sourceFilePath;
        this.desFilePath = desFilePath;
    }

    public SshInfo getSshInfo() {
        return sshInfo;
    }

    public void setSshInfo(SshInfo sshInfo) {
        this.sshInfo = sshInfo;
    }

    public String getSourceFilePath() {
        return sourceFilePath;
    }

    public void setSourceFilePath(String sourceFilePath) {
        this.sourceFilePath = sourceFilePath;
    }

    public String getDesFilePath() {
        return desFilePath;
    }

    public void setDesFilePath(String desFilePath) {
        this.desFilePath = desFilePath;
    }

    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }

    @Override
    public String toString() {
        String str = ToStringBuilder.reflectionToString(this, ToStringStyle.DEFAULT_STYLE);
        return str;
    }
}