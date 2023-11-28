package org.fedai.osx.broker.test.router;

import com.google.common.collect.Maps;
import okhttp3.*;
import org.fedai.osx.broker.pojo.RouterAddRequest;
import org.fedai.osx.broker.pojo.RouterTableGetRequest;
import org.fedai.osx.broker.pojo.SetSelfPartyRequest;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.utils.JsonUtil;
import org.junit.Test;

import java.io.IOException;
import java.util.Base64;
import java.util.HashSet;
import java.util.Map;

public class RouterSetTest {

    String url = "http://localhost:8807";

    @Test
    public void testGetRouteTable() {
        RouterTableGetRequest   getRequest = new RouterTableGetRequest();
        MediaType jsonMime = MediaType.parse("application/json");
        OkHttpClient okHttpClient = new OkHttpClient();
        Request.Builder builder = new Request.Builder();
        getRequest.setToken("1234");
        String requestContent = JsonUtil.object2Json(getRequest);
        System.err.println("request content "+requestContent);
        RequestBody requestBody = RequestBody.create(requestContent, jsonMime);
        Request request = builder.url(url + UriConstants.HTTP_GET_ROUTER).post(requestBody)
                .build();
        try {
            Response response = okHttpClient.newCall(request).execute();
            System.err.println(response);
            System.err.println("body " + new String(response.body().bytes()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    @Test
    public void testSetSelfParty() {
        SetSelfPartyRequest routerInfo = new SetSelfPartyRequest();
        HashSet  set = new HashSet<>();
        set.add("77377");
        routerInfo.setSelfParty(set);
        MediaType jsonMime = MediaType.parse("application/json");
        OkHttpClient okHttpClient = new OkHttpClient();
        Request.Builder builder = new Request.Builder();
        String requestContent = JsonUtil.object2Json(routerInfo);
        System.err.println("request content "+requestContent);
        RequestBody requestBody = RequestBody.create(requestContent, jsonMime);
        Request request = builder.url(url + UriConstants.HTTP_SET_SELF).post(requestBody)
                .build();
        try {
            Response response = okHttpClient.newCall(request).execute();
            System.err.println(response);
            System.err.println("body " + new String(response.body().bytes()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

//    curl --location 'http://localhost:8807/v1/inner/router/add' --header 'Content-Type: application/json' --data '{
//            "desPartyId": "1234",
//            "ip": "localhost",
//            "port": "9999"
//
//}'
//
//    curl --location 'http://localhost:8807/v1/inner/router/get' --header 'Content-Type: application/json'  --data '{
//}'
//
//    curl --location 'http://localhost:8807/v1/inner/router/self' --header 'Content-Type: application/json' --data '{
//            "selfParty": ["xxxxx"]
//}'





    @Test
    public void testAddRouter() {
        RouterAddRequest routerInfo = new RouterAddRequest();
        routerInfo.setDesPartyId("3479");
        routerInfo.setIp("localhost");
        routerInfo.setPort(8889);
        MediaType jsonMime = MediaType.parse("application/json");
        OkHttpClient okHttpClient = new OkHttpClient();
        Request.Builder builder = new Request.Builder();
//        header.forEach((k, v) -> {
//            builder.addHeader(k.toString(), v.toString());
//        });
        String requestContent = JsonUtil.object2Json(routerInfo);
        System.err.println("request content "+requestContent);
        RequestBody requestBody = RequestBody.create(requestContent, jsonMime);
        Request request = builder.url(url + UriConstants.HTTP_ADD_ROUTER).post(requestBody)
                .build();
        try {
            Response response = okHttpClient.newCall(request).execute();
            System.err.println(response);
            System.err.println("body " + new String(response.body().bytes()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
