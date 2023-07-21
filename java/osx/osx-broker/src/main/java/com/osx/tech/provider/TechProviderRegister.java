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

import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.frame.Lifecycle;
import com.osx.core.provider.TechProvider;
import com.osx.core.utils.ClassUtils;
import com.osx.core.utils.PropertiesUtil;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * 厂商选择
 */
public class TechProviderRegister implements Lifecycle {

    Logger logger  = LoggerFactory.getLogger(TechProviderRegister.class);
    ConcurrentMap<String, TechProvider> registerMap = new ConcurrentHashMap<>();
    final String  configFileName = "components/provider.properties";
    final
    public TechProvider select(String  techProviderCode ) {
        if(StringUtils.isEmpty(techProviderCode)){
            throw  new ParameterException("techProviderCode is null");
        }
        return this.registerMap.get(techProviderCode);
    }
    public void init() {
        Properties properties = PropertiesUtil.getProperties(MetaInfo.PROPERTY_CONFIG_DIR+Dict.SLASH+Dict.SLASH+configFileName);
        properties.forEach((k,v)->{
            try {
                this.registerMap.put(k.toString(), (TechProvider) ClassUtils.newInstance(v.toString()));
            }catch(Exception e){
                logger.error("provider {} class {} init error",k,v);
            }
        });
        logger.info("tech provider register : {}",this.registerMap);
    }

    @Override
    public void start() {
        init();
    }
    @Override
    public void destroy() {
        this.registerMap.clear();
    }

}



