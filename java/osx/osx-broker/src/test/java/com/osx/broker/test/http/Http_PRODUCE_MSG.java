package com.osx.broker.test.http;

import com.osx.broker.http.HttpClientPool;
import com.osx.broker.http.HttpsClientPool;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.utils.JsonUtil;
import org.junit.Test;
import org.ppc.ptp.Osx;

import java.util.HashMap;
import java.util.Map;

/**
 * @date 2023/5/29
 * @remark dd
 */
public class Http_PRODUCE_MSG {

    String url = "https://127.0.0.1:8088/osx/inbound";

    String caPath = "D:\\\\22\\\\ca.crt";
    String clientCertPath = "D:\\\\22\\\\174_1.crt";
    String clientKeyPath = "D:\\\\22\\\\174_1.key";

    @Test
    public void doHttp() {
        HttpClientPool.initPool();
        Osx.Outbound outbound = HttpClientPool.sendPtpPost(url, buildBody().getBytes(), buildHead());
        System.out.println(JsonUtil.formatJson(JsonUtil.object2Json(outbound.getPayload())));
    }

    @Test
    public void doHttpsSSL() {
        try {
            Osx.Outbound outbound = HttpsClientPool.sendPtpPost(url, buildBody().getBytes(), buildHead(), caPath, clientCertPath, clientKeyPath);
            System.out.println(JsonUtil.formatJson(JsonUtil.object2Json(outbound.getPayload())));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String buildBody() {
        Map<String, Object> body = new HashMap<>();
        body.put("uri", "/v2/partner/job/resource/apply");
        body.put("json_body", "{role=host, party_id=10008, job_id=202305251708508595320}");
        body.put("headers", "{}");
        body.put("method", "POST");
        body.put("MessageCode", "111");
        body.put("RetryCount", "111");
        return JsonUtil.object2Json(body);
    }

    public Map<String, String> buildHead() {
        Map<String, String> head = new HashMap<>();
//        CONSUME_MSG     ->  com.osx.broker.ptp.PtpConsumeService
//        APPLY_TOPIC     ->  com.osx.broker.ptp.PtpClusterTopicApplyService
//        APPLY_TOKEN     ->  com.osx.broker.ptp.PtpClusterTokenApplyService
//        QUERY_TOPIC     ->  com.osx.broker.ptp.PtpQueryTransferQueueService
//        PRODUCE_MSG     ->  com.osx.broker.ptp.PtpProduceService
//        ACK_MSG         ->  com.osx.broker.ptp.PtpAckService
//        UNARY_CALL      ->  com.osx.broker.ptp.PtpUnaryCallService
//        CANCEL_TOPIC    ->  com.osx.broker.ptp.PtpCancelTransferService
//        PUSH            ->  com.osx.broker.ptp.PtpPushService
        head.put("x-ptp-target-method", "PRODUCE_MSG");
        head.put("x-ptp-job-id", "202305251708508595320");
        head.put("x-ptp-tech-provider-code", "FATE");
        head.put("x-ptp-message-offset", "");
        head.put("x-ptp-source-inst-id", "");
        head.put("x-ptp-timestamp", "");
        head.put("x-ptp-target-component-name", "fateflow");
        head.put("x-ptp-message-topic", "");
        head.put("x-ptp-trace-id", "");
        head.put("x-ptp-source-node-id", "");
        head.put("x-ptp-source-method", "");
        head.put("x-ptp-token", "");
        head.put("x-ptp-message-flag", "");
        head.put("x-ptp-version", "");
        head.put("x-ptp-source-component-name", "");
        head.put("x-ptp-session-id", "");
        head.put("x-ptp-instance-id", "");
        head.put("x-ptp-target-node-id", "10008");
        head.put("x-ptp-target-inst-id", "");

        head.put(PtpHttpHeader.SessionID, "111");
        head.put(PtpHttpHeader.MessageTopic, "111");
        head.put(Osx.Metadata.MessageCode.name(), "111");
        head.put(Osx.Metadata.RetryCount.name(), "111");
        return head;
    }
}
