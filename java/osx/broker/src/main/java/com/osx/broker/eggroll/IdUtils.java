package com.osx.broker.eggroll;

import org.apache.commons.lang3.StringUtils;

public class IdUtils {
    private static String job = "job";
    private static String task = "task";

    public static String generateJobId(String sessionId,String tag, String delim){
        String result = String.join(delim, sessionId, "scala", job, TimeUtils.getNowMs(null));
        if (StringUtils.isBlank(tag)) {
            return result;
        }
        else{
            return  result+"_"+tag;
        }
            //s"${result}_${tag}"
    }



}



//
//    object IdUtils {
//private val job = "job"
//private val task = "task"
//        def generateJobId(sessionId: String, tag: String = "", delim: String = "-"): String = {
//        val result = String.join(delim, sessionId, "scala", job, TimeUtils.getNowMs())
//        if (StringUtils.isBlank(tag)) result else s"${result}_${tag}"
//        }
//
//        def generateTaskId(jobId: String, partitionId: Int, tag: String = "", delim: String = "-"): String =
//        if (StringUtils.isBlank(tag)) String.join(delim, jobId, task, partitionId.toString)
//        else String.join(delim, jobId, tag, task, partitionId.toString)
//        }