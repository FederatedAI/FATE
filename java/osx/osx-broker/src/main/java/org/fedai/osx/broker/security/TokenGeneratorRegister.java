package org.fedai.osx.broker.security;

import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.frame.Lifecycle;
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


public class TokenGeneratorRegister implements Lifecycle {

    Logger logger = LoggerFactory.getLogger(TokenGeneratorRegister.class);

    final String DEFAULT_KEY = "default";

    private Map<String,TokenGenerator>  tokenGeneratorMap =  new ConcurrentHashMap<>();


    @Override
    public void init() {
        if(MetaInfo.PROPERTY_OPEN_TOKEN_GENERATOR){
            String configFilePath= MetaInfo.PROPERTY_TOKEN_GENERATOR_CONFIG_PATH;
            File file = new File(configFilePath);
            Properties config = new Properties();
            try (InputStream inputStream = new BufferedInputStream(new FileInputStream(file))) {
                config.load(inputStream);
            }catch (Exception e){

            }
            config.forEach((k,v)->{
                if(v!=null){
                    try {
                        Class  genClass  =  Class.forName(v.toString());
                        Object rawObject = genClass.getConstructor().newInstance();
                        if(!(rawObject instanceof TokenGenerator)){
                            logger.error("create token generator err , {} ",v);
                            return ;
                        }
                        tokenGeneratorMap.put(k.toString(),(TokenGenerator)rawObject);
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException(e);
                    } catch (InvocationTargetException e) {
                        throw new RuntimeException(e);
                    } catch (InstantiationException e) {
                        throw new RuntimeException(e);
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    } catch (NoSuchMethodException e) {
                        throw new RuntimeException(e);
                    }
                }
            });
        }


    }

    @Override
    public void start() {
        logger.info("register token generator : {}",this.tokenGeneratorMap);
    }

    @Override
    public void destroy() {

    }
}
