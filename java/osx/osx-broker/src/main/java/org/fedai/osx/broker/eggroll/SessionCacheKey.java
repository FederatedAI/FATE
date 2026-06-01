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
package org.fedai.osx.broker.eggroll;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;

import java.util.Objects;

public class SessionCacheKey {

    private final String sessionId;
    private final String taskId;
    private final String rollSiteSessionId;
    private final String name;
    private final String tag;
    private final String srcRole;
    private final String srcPartyId;
    private final String dstRole;
    private final String dstPartyId;

    public SessionCacheKey(String sessionId, Proxy.Metadata metadata, ErRollSiteHeader rollSiteHeader) {
        this.sessionId = sessionId;
        this.taskId = metadata == null ? "" : metadata.getTask().getTaskId();
        this.rollSiteSessionId = rollSiteHeader.getRollSiteSessionId();
        this.name = rollSiteHeader.getName();
        this.tag = rollSiteHeader.getTag();
        this.srcRole = rollSiteHeader.getSrcRole();
        this.srcPartyId = rollSiteHeader.getSrcPartyId();
        this.dstRole = rollSiteHeader.getDstRole();
        this.dstPartyId = rollSiteHeader.getDstPartyId();
    }

    public SessionCacheKey(String sessionId, ErRollSiteHeader rollSiteHeader) {
        this(sessionId, Proxy.Metadata.getDefaultInstance(), rollSiteHeader);
    }

    public String getSessionId() {
        return sessionId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SessionCacheKey)) {
            return false;
        }
        SessionCacheKey that = (SessionCacheKey) o;
        return Objects.equals(sessionId, that.sessionId)
                && Objects.equals(taskId, that.taskId)
                && Objects.equals(rollSiteSessionId, that.rollSiteSessionId)
                && Objects.equals(name, that.name)
                && Objects.equals(tag, that.tag)
                && Objects.equals(srcRole, that.srcRole)
                && Objects.equals(srcPartyId, that.srcPartyId)
                && Objects.equals(dstRole, that.dstRole)
                && Objects.equals(dstPartyId, that.dstPartyId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sessionId, taskId, rollSiteSessionId, name, tag, srcRole, srcPartyId, dstRole, dstPartyId);
    }

    @Override
    public String toString() {
        return "SessionCacheKey{"
                + "sessionId='" + sessionId + '\''
                + ", taskId='" + taskId + '\''
                + ", rollSiteSessionId='" + rollSiteSessionId + '\''
                + ", name='" + name + '\''
                + ", tag='" + tag + '\''
                + ", srcRole='" + srcRole + '\''
                + ", srcPartyId='" + srcPartyId + '\''
                + ", dstRole='" + dstRole + '\''
                + ", dstPartyId='" + dstPartyId + '\''
                + '}';
    }
}
