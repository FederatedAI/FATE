package com.webank.ai.fate.serving.core.monitor;


import com.webank.ai.fate.serving.core.bean.Context;

import java.util.concurrent.atomic.AtomicLong;

public class WatchDog {

    private  static AtomicLong  RPC_IN_PROCESS =new  AtomicLong(0);

    public  static void  enter(Context context){
        RPC_IN_PROCESS.addAndGet(1);
    }

    public  static void  quit(Context  context){
        RPC_IN_PROCESS.decrementAndGet();
    }

    public  static long  get(){return RPC_IN_PROCESS.get();}

}
