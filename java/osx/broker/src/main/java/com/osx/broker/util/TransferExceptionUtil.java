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
