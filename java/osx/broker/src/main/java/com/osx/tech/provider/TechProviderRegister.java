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
package com.osx.tech.provider;

import com.google.common.base.Preconditions;
import com.osx.core.frame.Lifecycle;
import com.osx.core.provider.TechProvider;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * 厂商选择
 */
public class TechProviderRegister implements Lifecycle {

    ConcurrentMap<String, TechProvider> registerMap = new ConcurrentHashMap<>();

    //public TechProvider select(Pcp.Inbound inbound) {

        public TechProvider select(String  techProviderCode ) {
            Preconditions.checkArgument(techProviderCode != null);

        return this.registerMap.get(techProviderCode);
    }



    public void init() {
        FateTechProvider fateTechProvider = new FateTechProvider();
        fateTechProvider.init();
        this.registerMap.put(fateTechProvider.getProviderId(), fateTechProvider);
    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }


}



