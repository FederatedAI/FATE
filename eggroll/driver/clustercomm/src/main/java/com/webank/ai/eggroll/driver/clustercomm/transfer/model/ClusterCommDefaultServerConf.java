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

package com.webank.ai.eggroll.driver.clustercomm.transfer.model;

import com.webank.ai.eggroll.api.core.BasicMeta;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class ClusterCommDefaultServerConf {
    private List<BasicMeta.Endpoint> proxies;
/*
    public ClusterCommDefaultServerConf() {
        proxies = Lists.newArrayList();
    }

    public List<BasicMeta.Endpoint> getProxies() {
        return proxies;
    }

    public ClusterCommDefaultServerConf setProxies(List<BasicMeta.Endpoint> proxies) {
        this.proxies = proxies;
        return this;
    }

    public ClusterCommDefaultServerConf addProxyEndpoint(BasicMeta.Endpoint endpoint) {
        this.proxies.add(endpoint);
        return this;
    }*/
}
