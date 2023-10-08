package org.fedai.osx.broker.test.http;

import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import okhttp3.*;
import org.fedai.osx.broker.http.HttpClientPool;
import org.fedai.osx.broker.test.grpc.QueueTest;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.ptp.TargetMethod;
import org.junit.Before;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class HttpTest {

    Logger logger = LoggerFactory.getLogger(QueueTest.class);
//    String ip = "localhost";
//    //int port = 8250;//nginx
//    int port = 9889;//nginx
    String url="http://localhost:8089/osx/inbound";
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

        //HttpClientPool.initPool();
    }


    @Test
    public void   testConsumeMsg(){
        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
//        header.put(PtpHttpHeader.SourceNodeID,srcPartyId);
        header.put(PtpHttpHeader.TargetNodeID,desPartyId);
//        header.put(PtpHttpHeader.SourceInstID,"");
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
//        header.put(PtpHttpHeader.TargetMethod, TargetMethod.CONSUME_MSG.name());
        header.put(PtpHttpHeader.TargetComponentName,"");
        header.put(PtpHttpHeader.SourceComponentName,"");
        header.put(PtpHttpHeader.MessageTopic,transferId);
//        Osx.Message.Builder  messageBuilder = Osx.Message.newBuilder();
//        messageBuilder.setBody(ByteString.copyFrom("xiaoxiao1".getBytes(StandardCharsets.UTF_8)));
//        byte[]  content = messageBuilder.build().toByteArray();
       // System.err.println(HttpClientPool.sendPtpPost(url,null,header));


    }

    @Test
    public  void  testProduceMsg(){
        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.TargetNodeID,desPartyId);
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        header.put( PtpHttpHeader.Uri, UriConstants.PUSH);


        Osx.Message.Builder  messageBuilder = Osx.Message.newBuilder();
        messageBuilder.setBody(ByteString.copyFrom("xiaoxiao1".getBytes(StandardCharsets.UTF_8)));
        byte[]  content = messageBuilder.build().toByteArray();


        OkHttpClient  okHttpClient = new OkHttpClient();


        Request.Builder builder = new  Request.Builder();

        header.forEach((k,v)->{
            builder.addHeader(k.toString(),v.toString());
        });


        Transfer.RollSiteHeader.Builder  rollSiteHeader = Transfer.RollSiteHeader.newBuilder();
        rollSiteHeader.setDstRole("desRole");
        rollSiteHeader.setDstPartyId(desPartyId);
        rollSiteHeader.setSrcPartyId(srcPartyId);
        rollSiteHeader.setSrcRole("srcRole");
        rollSiteHeader.setRollSiteSessionId("testSessionId");
        Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
        packetBuilder.setHeader(Proxy.Metadata.newBuilder().setSeq(System.currentTimeMillis())
                .setSrc(Proxy.Topic.newBuilder().setPartyId(srcPartyId))
                .setDst(Proxy.Topic.newBuilder().setPartyId(desPartyId).setName("kaidengTestTopic").build())
                .setExt(rollSiteHeader.build().toByteString())

                .build());


//                Transfer.RollSiteHeader.Builder headerBuilder = Transfer.RollSiteHeader.newBuilder();
//                headerBuilder.setDstPartyId("10000");
        //   packetBuilder.setHeader(Proxy.Metadata.newBuilder().setExt(headerBuilder.build().toByteString()));
        Proxy.Data.Builder dataBuilder = Proxy.Data.newBuilder();
        dataBuilder.setKey("name");
        dataBuilder.setValue(ByteString.copyFrom(("xiaoxiao" ).getBytes()));
        packetBuilder.setBody(dataBuilder.build());

        Proxy.Packet packet = packetBuilder.build();

        Osx.PushInbound.Builder  pushInboundBuilder = Osx.PushInbound.newBuilder();
        pushInboundBuilder.setTopic("testtopic");
        pushInboundBuilder.setPayload(packet.toByteString());
        RequestBody requestBody = RequestBody.create(pushInboundBuilder.build().toByteArray());
        Request request =   builder.url(url).post(requestBody)
                .build();
        try {
            Response response =     okHttpClient.newCall(request).execute();
            response.body();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }


}
