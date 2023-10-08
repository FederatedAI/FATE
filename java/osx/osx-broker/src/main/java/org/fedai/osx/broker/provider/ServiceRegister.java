//package org.fedai.osx.broker.provider;
//
//import com.google.inject.Singleton;
//import org.fedai.osx.broker.service.ServiceRegisterInfo;
//import org.fedai.osx.core.constant.Dict;
//
//import java.util.HashSet;
//import java.util.List;
//import java.util.Set;
//import java.util.concurrent.ConcurrentHashMap;
//@Singleton
//public class ServiceRegister {
//
//    ConcurrentHashMap<String, Set<ServiceRegisterInfo>> registerMap= new ConcurrentHashMap<>();
//
//    public ServiceRegisterInfo get(String instId,
//                                   String nodeId,
//                                   String uri){
//
//        StringBuilder  sb = new StringBuilder();
//        sb.append(instId).append(Dict.SLASH).append(nodeId).append(Dict.SLASH).append(uri);
//
//        Set<ServiceRegisterInfo>  registerInfoSets =  registerMap.get(sb.toString());
//
//        if(registerInfoSets!=null){
//
//        }
//    }
//
//    public  void  register(ServiceRegisterInfo  serviceRegisterInfo){
//        String  registerKey = serviceRegisterInfo.buildRegisterKey();
//        if(registerMap.get(registerKey)==null){
//            registerMap.put(serviceRegisterInfo.buildRegisterKey(),new HashSet<ServiceRegisterInfo>());
//        }
//        registerMap.get(registerKey).add(serviceRegisterInfo);
//    }
//
//    public  void  unRegister(ServiceRegisterInfo  serviceRegisterInfo){
//        String  registerKey = serviceRegisterInfo.buildRegisterKey();
//        if(registerMap.contains(registerKey)){
//            registerMap.get(registerKey).remove(serviceRegisterInfo);
//        }
//    }
//
//
//
//
//}
