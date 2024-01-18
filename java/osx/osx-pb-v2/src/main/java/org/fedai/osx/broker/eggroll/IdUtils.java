/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.broker.eggroll;

import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.utils.TimeUtils;

public class IdUtils {
    private static String job = "job";
    private static String task = "task";

    public static String generateJobId(String sessionId, String tag, String delim) {
        String result = String.join(delim, sessionId, "scala", job, TimeUtils.getNowMs(null));
        if (StringUtils.isBlank(tag)) {
            return result;
        } else {
            return result + "_" + tag;
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