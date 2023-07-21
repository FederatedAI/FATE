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

package com.osx.api.router;
import com.osx.api.constants.Protocol;



public class RouterInfo {
    private Protocol protocol;
    private String sourcePartyId;
    private String desPartyId;
    private String desRole;
    private String sourceRole;
    private String url;
    private String host;
    private Integer port;
    private boolean useSSL = false;
    private String negotiationType;
    private String certChainFile;
    private String privateKeyFile;
    private String trustCertCollectionFile;
    private String caFile;
    private String version;

    public Protocol getProtocol() {
        return protocol;
    }

    public void setProtocol(Protocol protocol) {
        this.protocol = protocol;
    }

    public String getSourcePartyId() {
        return sourcePartyId;
    }

    public void setSourcePartyId(String sourcePartyId) {
        this.sourcePartyId = sourcePartyId;
    }

    public String getDesPartyId() {
        return desPartyId;
    }

    public void setDesPartyId(String desPartyId) {
        this.desPartyId = desPartyId;
    }

    public String getDesRole() {
        return desRole;
    }

    public void setDesRole(String desRole) {
        this.desRole = desRole;
    }

    public String getSourceRole() {
        return sourceRole;
    }

    public void setSourceRole(String sourceRole) {
        this.sourceRole = sourceRole;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public Integer getPort() {
        return port;
    }

    public void setPort(Integer port) {
        this.port = port;
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        this.useSSL = useSSL;
    }

    public String getNegotiationType() {
        return negotiationType;
    }

    public void setNegotiationType(String negotiationType) {
        this.negotiationType = negotiationType;
    }

    public String getCertChainFile() {
        return certChainFile;
    }

    public void setCertChainFile(String certChainFile) {
        this.certChainFile = certChainFile;
    }

    public String getPrivateKeyFile() {
        return privateKeyFile;
    }

    public void setPrivateKeyFile(String privateKeyFile) {
        this.privateKeyFile = privateKeyFile;
    }

    public String getTrustCertCollectionFile() {
        return trustCertCollectionFile;
    }

    public void setTrustCertCollectionFile(String trustCertCollectionFile) {
        this.trustCertCollectionFile = trustCertCollectionFile;
    }

    public String getCaFile() {
        return caFile;
    }

    public void setCaFile(String caFile) {
        this.caFile = caFile;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public boolean isCycle() {
        return isCycle;
    }

    public void setCycle(boolean cycle) {
        isCycle = cycle;
    }

    private boolean isCycle;

    public String toKey() {
        StringBuffer sb = new StringBuffer();
        if(Protocol.grpc.equals(protocol)) {
            sb.append(host).append("_").append(port);
            if (negotiationType != null)
                sb.append("_").append(negotiationType);
        }else {
            sb.append(url);
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return  toKey();
    }

    public String getResource() {
        StringBuilder sb = new StringBuilder();
        sb.append(sourcePartyId).append("-").append(desPartyId);
        return sb.toString();
    }


}