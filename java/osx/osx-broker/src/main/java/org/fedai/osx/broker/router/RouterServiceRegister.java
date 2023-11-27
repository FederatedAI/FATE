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
package org.fedai.osx.broker.router;

import com.google.inject.Inject;
import com.google.inject.Injector;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.core.service.ApplicationStartedRunner;
import org.fedai.osx.core.utils.PropertiesUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * 厂商选择
 */
@Singleton
public class RouterServiceRegister implements Lifecycle, ApplicationStartedRunner {

    final String configFileName = "components/router.properties";
    Logger logger = LoggerFactory.getLogger(RouterServiceRegister.class);
    @Inject
    Injector injector;
    ConcurrentMap<String, RouterService> registerMap = new ConcurrentHashMap<>();

    final public RouterService select(OsxContext fateContext) {
        if (StringUtils.isEmpty(fateContext.getTechProviderCode())) {
            fateContext.setTechProviderCode(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        }
        return this.select(fateContext.getTechProviderCode());
    }

    final public RouterService select(String techProviderCode) {
        if(StringUtils.isEmpty(techProviderCode)){
            techProviderCode = MetaInfo.PROPERTY_FATE_TECH_PROVIDER;
        }
        if (this.registerMap.containsKey(techProviderCode)) {
            return this.registerMap.get(techProviderCode);
        } else {
            return this.registerMap.get(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        }
    }

    public RouterService getTechProvider(OsxContext context) {
        RouterService routerService = this.select(context.getTechProviderCode());
        if (routerService == null) {
            routerService = this.select("default");
        }
        return routerService;
    }


    public void init() {
        Properties properties = PropertiesUtil.getProperties(MetaInfo.PROPERTY_CONFIG_DIR + Dict.SLASH + Dict.SLASH + configFileName);
        properties.forEach((k, v) -> {
            try {
                this.registerMap.put(k.toString(), (RouterService) injector.getInstance(Class.forName(v.toString())));
            } catch (Exception e) {
                logger.error("router {} class {} init error", k, v, e);
            }
        });
        logger.info("router service register : {}", this.registerMap);
    }

    @Override
    public void start() {
        init();
    }

    @Override
    public void destroy() {
        this.registerMap.clear();
    }

    @Override
    public void run(String[] args) throws Exception {
        start();
    }
}



