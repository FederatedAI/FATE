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

package org.fedai.osx.api.router;
import lombok.Data;


@Data
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
    private String instId;
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