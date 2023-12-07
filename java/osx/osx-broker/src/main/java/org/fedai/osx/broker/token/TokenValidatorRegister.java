package org.fedai.osx.broker.token;

import com.google.inject.Inject;
import com.google.inject.Injector;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.frame.Lifecycle;
import org.fedai.osx.core.service.ApplicationStartedRunner;
import org.fedai.osx.core.utils.PropertiesUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
@Singleton
public class TokenValidatorRegister implements Lifecycle , ApplicationStartedRunner {

    final String configFileName = "components/token_validator.properties";

    Logger logger = LoggerFactory.getLogger(TokenValidatorRegister.class);

    @Inject
    Injector injector;

    ConcurrentMap<String, TokenValidator> registerMap = new ConcurrentHashMap<>();

    final public TokenValidator select(String name) {
        if (this.registerMap.containsKey(name)) {
            return this.registerMap.get(name);
        }
        return null;
    }

    public void init() {
        Properties properties = PropertiesUtil.getProperties(MetaInfo.PROPERTY_CONFIG_DIR + Dict.SLASH + Dict.SLASH + configFileName);
        properties.forEach((k, v) -> {
            try {
                if(k.toString().contains(".impl")){
                    TokenValidator  tokenValidator = (TokenValidator) injector.getInstance(Class.forName(v.toString()));
                    String name = k.toString().split("\\.")[0];
                    tokenValidator.init(name,properties);
                    this.registerMap.put(name, tokenValidator);
                }
            } catch (Exception e) {
                logger.error("token validator {} class {} init error", k, v, e);
            }
        });
        logger.info("token validator register : {}", this.registerMap);
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
