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

package com.webank.ai.fate.networking.proxy.security;

import io.grpc.netty.shaded.io.netty.handler.ssl.util.SimpleTrustManagerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.net.ssl.ManagerFactoryParameters;
import javax.net.ssl.TrustManager;
import java.security.KeyStore;

@Component
@Scope("prototype")
public class SimpleTrustAllCertsManagerFactory extends SimpleTrustManagerFactory {
    @Autowired
    private TrustAllCertsManager trustAllCertsManager;

    @Override
    protected void engineInit(KeyStore keyStore) throws Exception {

    }

    @Override
    protected void engineInit(ManagerFactoryParameters managerFactoryParameters) throws Exception {

    }

    @Override
    protected TrustManager[] engineGetTrustManagers() {
        return new TrustManager[]{trustAllCertsManager};
    }
}
