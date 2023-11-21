//package org.fedai.osx.broker.router;
//
//import com.google.inject.Singleton;
//import org.fedai.osx.core.config.MetaInfo;
//import org.fedai.osx.core.constant.Dict;
//import org.fedai.osx.core.frame.Lifecycle;
//import org.fedai.osx.core.service.ApplicationStartedRunner;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.BufferedInputStream;
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.InputStream;
//import java.util.Properties;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.concurrent.ConcurrentMap;
//@Singleton
//public class RouterRegister implements Lifecycle  , ApplicationStartedRunner {
//
//    Logger logger = LoggerFactory.getLogger(RouterRegister.class);
//
//    private final String ROUTER_CONFIG_FILE = "components/router.properties";
//
//    private ConcurrentMap<String,RouterService>  routerServiceMap = new ConcurrentHashMap<>();
//
//    public RouterService  getRouterService(String  key){
//        return routerServiceMap.get(key);
//    }
//
//    @Override
//    public void init() {
//        String configDir= MetaInfo.PROPERTY_CONFIG_DIR;
//        String fileName = configDir+ Dict.SLASH+ROUTER_CONFIG_FILE;
//        File file = new File(fileName);
//        Properties config = new Properties();
//        try (InputStream inputStream = new BufferedInputStream(new FileInputStream(file))) {
//            config.load(inputStream);
//        }catch (Exception e){
//            logger.error("can not found {}",fileName);
//        }
//        config.forEach((k,v)->{
//            if(v!=null){
//                try {
//                    Class  genClass  =  Class.forName(v.toString());
//                    Object rawObject =  genClass.getConstructor().newInstance();
//                    routerServiceMap.put(k.toString(),(RouterService)rawObject);
//                    if(rawObject instanceof Lifecycle){
//                        ( (Lifecycle)rawObject).init();
//                    }
//                } catch (Exception e) {
//                   logger.error("register router error {} : {}",k,v,e);
//                }
//            }
//        });
//    }
//
//    @Override
//    public void start() {
//        routerServiceMap.forEach((k,v)->{
//            if(v instanceof Lifecycle){
//                ( (Lifecycle)v).start();
//            }
//        });
//        logger.info("router register : {}",routerServiceMap);
//    }
//
//    @Override
//    public void destroy() {
//
//    }
//
//    @Override
//    public void run(String[] args) throws Exception {
//        init();
//    }
//}
