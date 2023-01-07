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
package com.osx.broker.util;

public class TransferExceptionUtil {


}


//    private def genExceptionDescription(t: Throwable, topic: Topic = null): String = {
//        val locMsg = t.getLocalizedMessage
//        val stackInfo = ExceptionUtils.getStackTrace(t)
//        val myPartyId = RollSiteConfKeys.EGGROLL_ROLLSITE_PARTY_ID.get()
//        var desc = s"Error from partyId=${myPartyId}:\n-------------\n"
//        val host = RollSiteConfKeys.EGGROLL_ROLLSITE_HOST.get()
//        if (locMsg != null && locMsg.contains("[Roll Site Error TransInfo]")) {
//        desc = s"${locMsg} --> ${host}(${myPartyId})"
//        } else {
//        desc = f"\n[Roll Site Error TransInfo] \n location msg=${locMsg} \n stack info=${stackInfo} \n"
//        if (topic != null) {
//        val locationInfo = f"\nlocationInfo: topic.getName=${topic.getName} " +
//        f"topic.getPartyId=${topic.getPartyId}"
//        desc = desc + locationInfo
//        }
//        desc = desc + f"\nexception trans path: ${host}(${myPartyId})"
//        }
//        desc
//        }


//    def throwableToException(t: Throwable, topic: Topic = null): StatusRuntimeException = {
//        //if (t.isInstanceOf[StatusRuntimeException]) return t.asInstanceOf[StatusRuntimeException]
//        val e = if (t != null) {
//        t
//        } else {
//        new IllegalStateException(s"t is null when throwing exception. myPartyId=${RollSiteConfKeys.EGGROLL_ROLLSITE_PARTY_ID.get()}")
//        }
//        val desc = genExceptionDescription(e, topic)
//        val status = Status.fromThrowable(e).withDescription(desc)
//        status.asRuntimeException()
//        }
