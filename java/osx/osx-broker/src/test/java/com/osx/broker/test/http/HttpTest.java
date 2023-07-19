package com.osx.broker.test.http;

import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.osx.broker.http.HttpClientPool;
import com.osx.broker.test.grpc.QueueTest;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.ptp.TargetMethod;
import io.grpc.ManagedChannel;
import org.junit.Before;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.Map;

public class HttpTest {

    Logger logger = LoggerFactory.getLogger(QueueTest.class);
//    String ip = "localhost";
//    //int port = 8250;//nginx
//    int port = 9889;//nginx
    String url="http://localhost:8222/osx/inbound";
    String desPartyId = "10000";
    String desRole = "";
    String srcPartyId = "9999";
    String srcRole = "";
    String transferId = "testTransferId";
    String sessionId = "testSessionId";


    @Before
    public void init() {

        MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT = Integer.valueOf("500");
        MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_TIME_OUT = Integer.valueOf("2000");
        MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_SOCK_TIME_OUT = Integer.valueOf(3000);
        MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_MAX_TOTAL = Integer.valueOf(500);
        MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_DEF_MAX_PER_ROUTE = Integer.valueOf(200);
//        MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_SOCK_TIME_OUT = Integer.valueOf(10000);
//        MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_CONN_TIME_OUT = Integer.valueOf(10000);
//        MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_CONN_REQ_TIME_OUT = Integer.valueOf(10000);
//        MetaInfo.PROPERTY_HTTP_CLIENT_TRAN_CONN_REQ_TIME_OUT = Integer.valueOf(60000);
//        MetaInfo.PROPERTY_HTTP_CLIENT_TRAN_CONN_TIME_OUT = Integer.valueOf(60000);
//        MetaInfo.PROPERTY_HTTP_CLIENT_TRAN_SOCK_TIME_OUT = Integer.valueOf(60000);

        HttpClientPool.initPool();
    }


    @Test
    public void   testConsumeMsg(){
        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.SourceNodeID,srcPartyId);
        header.put(PtpHttpHeader.TargetNodeID,desPartyId);
        header.put(PtpHttpHeader.SourceInstID,"");
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        header.put(PtpHttpHeader.TargetMethod, TargetMethod.CONSUME_MSG.name());
        header.put(PtpHttpHeader.TargetComponentName,"");
        header.put(PtpHttpHeader.SourceComponentName,"");
        header.put(PtpHttpHeader.MessageTopic,transferId);
//        Osx.Message.Builder  messageBuilder = Osx.Message.newBuilder();
//        messageBuilder.setBody(ByteString.copyFrom("xiaoxiao1".getBytes(StandardCharsets.UTF_8)));
//        byte[]  content = messageBuilder.build().toByteArray();
        System.err.println(HttpClientPool.sendPtpPost(url,null,header));


    }

    @Test
    public  void  testProduceMsg(){
        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.SourceNodeID,desPartyId);//desPartyId
        header.put(PtpHttpHeader.TargetNodeID,srcPartyId);//srcPartyId
        header.put(PtpHttpHeader.SourceInstID,"");
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        header.put(PtpHttpHeader.TargetMethod, TargetMethod.PRODUCE_MSG.name());
        header.put(PtpHttpHeader.TargetComponentName,"");
        header.put(PtpHttpHeader.SourceComponentName,"");
        header.put(PtpHttpHeader.MessageTopic,transferId);

        Osx.Message.Builder  messageBuilder = Osx.Message.newBuilder();
        messageBuilder.setBody(ByteString.copyFrom("xiaoxiao1".getBytes(StandardCharsets.UTF_8)));
        byte[]  content = messageBuilder.build().toByteArray();
        System.err.println(HttpClientPool.sendPtpPost(url,content,header));

    }


}
