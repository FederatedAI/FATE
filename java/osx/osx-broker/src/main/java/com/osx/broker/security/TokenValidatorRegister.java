package com.osx.broker.security;

import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.frame.Lifecycle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

public class TokenValidatorRegister implements Lifecycle {

    Logger logger = LoggerFactory.getLogger(TokenValidatorRegister.class);

    final String DEFAULT_KEY = "default";
    final String TOKEY_VALIDATOR_CONFIG_FILE="token_validator.properties";

    private Map<String,TokenValidator> tokenValidatorMap =  new ConcurrentHashMap<>();

    public TokenValidator  getTokenValidator(String key,String defaultKey){
       TokenValidator result = tokenValidatorMap.get(key);
        if(result ==null){
            result =  tokenValidatorMap.get(defaultKey);
        };
        return  result;
    }
    @Override
    public void init() {
        if(MetaInfo.PROPERTY_OPEN_TOKEN_GENERATOR){
            String configDir= MetaInfo.PROPERTY_CONFIG_DIR;
            String fileName = configDir+ Dict.SLASH+TOKEY_VALIDATOR_CONFIG_FILE;
            File file = new File(fileName);
            Properties config = new Properties();
            try (InputStream inputStream = new BufferedInputStream(new FileInputStream(file))) {
                config.load(inputStream);
            }catch (Exception e){
                logger.error("parse file {} error",fileName);
            }

            config.forEach((k,v)->{
                if(v!=null){
                    try {
                        Class  genClass  =  Class.forName(v.toString());
                        Object rawObject = genClass.getConstructor().newInstance();
                        if(!(rawObject instanceof TokenValidator)){
                            logger.error("parse token validator err , {} ",v);
                            return ;
                        }
                        tokenValidatorMap.put(k.toString(),(TokenValidator)rawObject);
                    } catch (Exception e) {
                        logger.error("register token validator error {} : {}",k,v);
                    }
                }
            });
        }
    }

    @Override
    public void start() {
        logger.info("register token validator : {}",this.tokenValidatorMap);
    }

    @Override
    public void destroy() {
        this.tokenValidatorMap.clear();
    }


}
