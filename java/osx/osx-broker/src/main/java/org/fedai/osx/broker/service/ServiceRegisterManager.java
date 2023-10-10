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
package org.fedai.osx.broker.service;

import com.google.common.collect.Lists;
import com.google.inject.Inject;
import com.google.inject.Injector;
import com.google.inject.Singleton;
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.constants.ServiceType;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.InvalidRequestException;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.ApplicationStartedRunner;
import org.reflections.Reflections;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Singleton
@Slf4j
public class ServiceRegisterManager implements ApplicationStartedRunner {

    ConcurrentHashMap<String , List<ServiceRegisterInfo>>   serviceRegisterMap = new ConcurrentHashMap<>();

    @Inject
    Injector injector;

    public   void  register(ServiceRegisterInfo  serviceRegisterInfo){
        String key = serviceRegisterInfo.buildRegisterKey();
        if(serviceRegisterMap.get(key)==null){
            serviceRegisterMap.put(serviceRegisterInfo.buildRegisterKey(), Lists.newArrayList());
        }
        serviceRegisterMap.get(key).add(serviceRegisterInfo);
        log.info("register service {}",key);
    }

    public  ServiceRegisterInfo  getServiceWithLoadBalance(OsxContext  osxContext,String node, String uri,boolean  interInvoke){
        long now = System.currentTimeMillis();
        ServiceRegisterInfo  result = null;
        String  key = ServiceRegisterInfo.buildKey(node,uri);
//        log.info("=======query key {}",key);
        List<ServiceRegisterInfo>  services =  serviceRegisterMap.get(key );
        if(services!=null&&services.size()>0){
            result =   services.get((int)(now% services.size()));
            if(interInvoke&&result.isAllowInterUse()){
                throw  new InvalidRequestException();
            }

        }
        return  result;
    }



    @Override
    public void run(String[] args) throws Exception {
        Reflections  reflections = new Reflections("org.fedai.osx.broker");
        Set<Class<?>> classes = reflections.getTypesAnnotatedWith(Register.class);
        classes.forEach(clazz->{
                MetaInfo.PROPERTY_SELF_PARTY.forEach(partyId->{

                    Register  register = clazz.getAnnotation(Register.class);
                    String [] uris = register.uris();
                    for(String uri:uris) {
                        ServiceRegisterInfo serviceRegisterInfo = new ServiceRegisterInfo();
                        serviceRegisterInfo.setUri(uri);
                        serviceRegisterInfo.setNodeId(partyId);
//                        serviceRegisterInfo.setProtocol(register.protocol());
                        serviceRegisterInfo.setServiceType(ServiceType.inner);
                        AbstractServiceAdaptorNew serviceAdaptor = (AbstractServiceAdaptorNew) injector.getInstance(clazz);
                        serviceRegisterInfo.setServiceAdaptor(serviceAdaptor);
                        this.register(serviceRegisterInfo);
                    }
                });

        });
    }
}
