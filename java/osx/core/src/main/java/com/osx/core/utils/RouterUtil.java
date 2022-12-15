package com.osx.core.utils;

public class RouterUtil {


    //forward
//    public  String getSourceName1(){
//        return "receive_"+source;
//    }
//    public  String getSourceName2(){
//        return "to_"+source;
//    }
//    public  String getSourceName3(){
//        return "to_"+destination;
//    }
//    public  String getSourceName4(){
//        return "receive_"+destination;
//    }

    public  static  String  getReceiveKey(String  key){
        return "receive_"+key;
    }
    public  static  String  getSendKey(String key){
        return  "to_"+key;
    }
}
