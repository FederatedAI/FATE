package org.fedai.osx.broker.test.router;

import com.google.common.collect.Maps;
import okhttp3.*;
import org.fedai.osx.broker.pojo.RouterAddRequest;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.utils.JsonUtil;
import org.junit.Test;

import java.io.IOException;
import java.util.Base64;
import java.util.Map;

public class RouterSetTest {

    String url = "http://localhost:8807";
    @Test
    public void testSetRouter() {


        RouterAddRequest routerInfo = new RouterAddRequest();
        routerInfo.setDesPartyId("3477");
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
