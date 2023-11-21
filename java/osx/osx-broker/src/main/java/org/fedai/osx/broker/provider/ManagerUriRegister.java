package org.fedai.osx.broker.provider;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.service.ServiceRegisterInfo;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.core.utils.ClassUtils;
import org.fedai.osx.core.utils.PropertiesUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
@Slf4j
public class ManagerUriRegister implements Lifecycle {




    ConcurrentMap<String, ServiceRegisterInfo> registerMap = new ConcurrentHashMap<>();
    final String  configFileName = "components/manager_uri.properties";


    public ServiceRegisterInfo  getServiceRegisterInfo(String uri){
        return registerMap.get(uri);
    }


    public void init() {
        Properties properties = PropertiesUtil.getProperties(MetaInfo.PROPERTY_CONFIG_DIR+ Dict.SLASH+Dict.SLASH+configFileName);
        properties.forEach((k,v)->{
            try {
                ServiceRegisterInfo  serviceRegisterInfo = new  ServiceRegisterInfo();
                serviceRegisterInfo.setUri(k.toString());
                this.registerMap.put(k.toString(),serviceRegisterInfo);
            }catch(Exception e){
                log.error("provider {} class {} init error",k,v);
            }
        });
        log.info("tech provider register : {}",this.registerMap);
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
