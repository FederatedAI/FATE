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

package com.webank.ai.fate.core.network.http.client;

import com.webank.ai.fate.core.utils.ObjectTransform;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.config.Registry;
import org.apache.http.config.RegistryBuilder;
import org.apache.http.conn.socket.ConnectionSocketFactory;
import org.apache.http.conn.socket.PlainConnectionSocketFactory;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustSelfSignedStrategy;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.DefaultHttpRequestRetryHandler;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.protocol.HttpClientContext;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class HttpClientPool {
    private static final Logger LOGGER = LogManager.getLogger();
    private static PoolingHttpClientConnectionManager poolConnManager;
    private static RequestConfig requestConfig;
    private static CloseableHttpClient httpClient;

    private static void config(HttpRequestBase httpRequestBase, Map<String, String> headers) {
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(10 * 1000)
                .setConnectTimeout(10 * 1000)
                .setSocketTimeout(10 * 1000).build();
        httpRequestBase.addHeader("Content-Type", "application/json;charset=UTF-8");
        if (headers != null){
            headers.forEach((key, value) -> {
                httpRequestBase.addHeader(key, value);
            });
        }
        httpRequestBase.setConfig(requestConfig);
    }

    public static void initPool() {
        try {
            SSLContextBuilder builder = new SSLContextBuilder();
            builder.loadTrustMaterial(null, new TrustSelfSignedStrategy());
            SSLConnectionSocketFactory sslsf = new SSLConnectionSocketFactory(builder.build());
            Registry<ConnectionSocketFactory> socketFactoryRegistry = RegistryBuilder.<ConnectionSocketFactory>create().register(
                    "http", PlainConnectionSocketFactory.getSocketFactory()).register(
                    "https", sslsf).build();
            poolConnManager = new PoolingHttpClientConnectionManager(
                    socketFactoryRegistry);
            poolConnManager.setMaxTotal(500);
            poolConnManager.setDefaultMaxPerRoute(200);
            int socketTimeout = 10000;
            int connectTimeout = 10000;
            int connectionRequestTimeout = 10000;
            requestConfig = RequestConfig.custom().setConnectionRequestTimeout(
                    connectionRequestTimeout).setSocketTimeout(socketTimeout).setConnectTimeout(
                    connectTimeout).build();
            httpClient = getConnection();
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException ex) {
            LOGGER.error("init http client pool failed:", ex);
        }
    }

    private static CloseableHttpClient getConnection() {
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(poolConnManager)
                .setDefaultRequestConfig(requestConfig)
                .setRetryHandler(new DefaultHttpRequestRetryHandler(0, false))
                .build();
        return httpClient;
    }

    public static String post(String url, Map<String, Object> requestData){
        return sendPost(url, requestData, null);
    }

    public static String post(String url, Map<String, Object> requestData, Map<String, String> headers){
        return sendPost(url, requestData, headers);
    }

    public static String sendPost(String url, Map<String, Object> requestData, Map<String, String> headers){
        HttpPost httpPost = new HttpPost(url);
        config(httpPost, headers);
        StringEntity stringEntity = new StringEntity(ObjectTransform.bean2Json(requestData), "UTF-8");
        stringEntity.setContentEncoding("UTF-8");
        httpPost.setEntity(stringEntity);
        return getResponse(httpPost);
    }

    public static String get(String url, Map<String, String> headers) {
        return sendGet(url, headers);
    }

    public static String get(String url) {
        return sendGet(url, null);
    }

    public static String sendGet(String url, Map<String, String> headers) {
        HttpGet httpGet = new HttpGet(url);
        config(httpGet, headers);
        return getResponse(httpGet);
    }

    private static String getResponse(HttpRequestBase request) {
        CloseableHttpResponse response = null;
        try {
            response = httpClient.execute(request,
                    HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            String result = EntityUtils.toString(entity, "utf-8");
            EntityUtils.consume(entity);
            return result;
        } catch (IOException ex) {
            LOGGER.error("get http response failed:", ex);
            return null;
        } finally {
            try {
                if (response != null)
                    response.close();
            } catch (IOException ex) {
                LOGGER.error("get http response failed:", ex);
            }
        }
    }
}
