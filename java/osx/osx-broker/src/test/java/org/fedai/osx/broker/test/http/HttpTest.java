package org.fedai.osx.broker.test.http;

import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;

import okhttp3.*;
import org.fedai.osx.broker.http.HttpClientPool;
import org.fedai.osx.broker.test.grpc.QueueTest;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.ptp.TargetMethod;
import org.fedai.osx.core.utils.JsonUtil;
import org.junit.Before;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class HttpTest {

    Logger logger = LoggerFactory.getLogger(QueueTest.class);
//    String ip = "localhost";
//    //int port = 8250;//nginx
//    int port = 9889;//nginx
    String url="http://localhost:7304";
    String desPartyId = "9999";
    String desRole = "";
    String srcPartyId = "9999";
    String srcRole = "";
    String transferId = "testTransferId";
    String sessionId = "testSessionId111";

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
    public  void  testPopMsg(){
        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.TargetNodeID,desPartyId);
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        MediaType jsonMime=MediaType.parse("application/json");
        OkHttpClient  okHttpClient = new OkHttpClient.Builder().connectTimeout(15, TimeUnit.SECONDS).
                readTimeout(10,TimeUnit.MINUTES).writeTimeout(10,TimeUnit.MINUTES).build();

        Request.Builder builder = new  Request.Builder();
        header.forEach((k,v)->{
            builder.addHeader(k.toString(),v.toString());
        });
        Base64.Encoder  encoder = Base64.getEncoder();
        Map sendBodyData= Maps.newHashMap();
        sendBodyData.put("topic","testTopic10000");
//        sendBodyData.put("payload",new String(encoder.encode("my name is world".getBytes(StandardCharsets.UTF_8))));
        RequestBody requestBody = RequestBody.create(JsonUtil.object2Json(sendBodyData),jsonMime);
        Request request =   builder.url(url+UriConstants.HTTP_POP).post(requestBody)
                .build();
        try {
            Response response =     okHttpClient.newCall(request).execute();

            Map result = JsonUtil.json2Object(response.body().string(),Map.class);
            System.err.println(result);
            Base64.Decoder decoder = Base64.getDecoder();
            if(result.get("payload")!=null){
              System.err.println( "payload : "+ result.get("payload").toString());
              System.err.println( "ori msg : "+ new String(decoder.decode(result.get("payload").toString())));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Test
    public  void  testPeekMsg(){

        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.TargetNodeID,desPartyId);
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        header.put( PtpHttpHeader.Uri, UriConstants.PUSH);
        MediaType jsonMime=MediaType.parse("application/json");
        OkHttpClient  okHttpClient = new OkHttpClient();
        Request.Builder builder = new  Request.Builder();
        header.forEach((k,v)->{
            builder.addHeader(k.toString(),v.toString());
        });
        Base64.Encoder  encoder = Base64.getEncoder();
        Map sendBodyData= Maps.newHashMap();
        sendBodyData.put("topic","testTopic10000");
//        sendBodyData.put("payload",new String(encoder.encode("my name is world".getBytes(StandardCharsets.UTF_8))));
        RequestBody requestBody = RequestBody.create(JsonUtil.object2Json(sendBodyData),jsonMime);
        Request request =   builder.url(url+UriConstants.HTTP_PEEK).post(requestBody)
                .build();
        try {
            Response response =     okHttpClient.newCall(request).execute();
            System.err.println(response);
            System.err.println("body " +new String(response.body().bytes()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    @Test
    public  void  testPush(){

//        curl --location 'http://127.0.0.1:7304/v1/interconn/chan/push' --header 'Content-Type: application/json' --header 'x-ptp-target-node-id: 123' --header 'x-ptp-session-id: 1' --data '{
//        "topic": "0",
//                "metadata": {
//            "a": "b"
//        },
//        "payload": "MTkyLjE2OC4xMDAuNjM6MTczMDQ="
//    }'

        Map header = Maps.newHashMap();
        header.put(PtpHttpHeader.Version,"");
        header.put(PtpHttpHeader.TechProviderCode, MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        header.put(PtpHttpHeader.Token,"");
        header.put(PtpHttpHeader.TargetNodeID,srcPartyId);
        header.put(PtpHttpHeader.TargetInstID,"");
        header.put(PtpHttpHeader.SessionID,sessionId);
        header.put( PtpHttpHeader.Uri, UriConstants.PUSH);
        MediaType jsonMime=MediaType.parse("application/json");
        OkHttpClient  okHttpClient = new OkHttpClient();
        Request.Builder builder = new  Request.Builder();
        header.forEach((k,v)->{
            builder.addHeader(k.toString(),v.toString());
        });
        Base64.Encoder  encoder = Base64.getEncoder();
        Map sendBodyData= Maps.newHashMap();
        sendBodyData.put("topic","testTopic10000");
        sendBodyData.put("payload",new String(encoder.encode("my name is world".getBytes(StandardCharsets.UTF_8))));
        System.err.println("send : "+sendBodyData);

        RequestBody requestBody = RequestBody.create(JsonUtil.object2Json(sendBodyData),jsonMime);
        Request request =   builder.url(url+UriConstants.HTTP_PUSH).post(requestBody)
                .build();
        try {
            Response response =     okHttpClient.newCall(request).execute();
            System.err.println(response);
            System.err.println("body" +new String(response.body().bytes()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public  static  void main(String[] args){
        HttpTest  httpTest = new HttpTest();
        httpTest.testPopMsg();
    }


}
