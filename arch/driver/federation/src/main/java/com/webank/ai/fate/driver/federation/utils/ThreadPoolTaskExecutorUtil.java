package com.webank.ai.fate.driver.federation.utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;

import java.util.concurrent.Callable;


public class ThreadPoolTaskExecutorUtil {

    private static Logger LOGGER = LogManager.getLogger();

    public static <T> ListenableFuture<T>  submitListenable(ThreadPoolTaskExecutor executor, Callable<T> callable, int[] sleepTimes, int[] tryCount){

        ListenableFuture<T> resultListenableFuture = null;

            try {
                resultListenableFuture = executor.submitListenable(callable);
            }catch(TaskRejectedException e){
                boolean  success = false;
                for(int i=0;i<sleepTimes.length&&i<tryCount.length;i++) {
                        int  tryNum =  tryCount[i];
                        int  sleepTime = sleepTimes[i];
                        int  count = 0;
                        do {
                            count++;
                            try {
                                try {
                                    Thread.sleep(sleepTime);
                                } catch (InterruptedException e1) {
                                    e1.printStackTrace();
                                }
                                resultListenableFuture = executor.submitListenable(callable);
                                return resultListenableFuture;
                            } catch (TaskRejectedException taskException) {
                                LOGGER.error("[FEDERATION][THREADPOOL] submit exception,sleep {} tryCount {}",sleepTime,count);
                            }
                        }while(count<tryNum);
                }
                if(resultListenableFuture==null) {
                    throw e;
                }
            }
            return  resultListenableFuture;
    }


    public static ListenableFuture<?>  submitListenable(ThreadPoolTaskExecutor executor, Runnable callable, int[] sleepTimes, int[] tryCount){

        ListenableFuture<?> resultListenableFuture = null;

        try {
            resultListenableFuture = executor.submitListenable(callable);
        }catch(TaskRejectedException e){
            for(int i=0;i<sleepTimes.length&&i<tryCount.length;i++) {
                int  tryNum =  tryCount[i];
                int  sleepTime = sleepTimes[i];
                int  count = 0;
                do {
                    count++;
                    try {
                        try {
                            Thread.sleep(sleepTime);
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                        resultListenableFuture = executor.submitListenable(callable);
                        return resultListenableFuture;
                    } catch (TaskRejectedException taskException) {
                        LOGGER.error("[FEDERATION][THREADPOOL] submit exception,sleep {} tryCount {}",sleepTime,count);
                    }
                }while(count<tryNum);
            }
            if(resultListenableFuture==null) {
                throw e;
            }
        }
        return  resultListenableFuture;
    }
}
