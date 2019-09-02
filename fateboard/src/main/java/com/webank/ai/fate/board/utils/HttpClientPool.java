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
package com.webank.ai.fate.board.utils;

import com.alibaba.fastjson.JSON;
import org.apache.commons.io.Charsets;
import org.apache.http.HttpEntity;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.client.protocol.HttpClientContext;
import org.apache.http.config.Registry;
import org.apache.http.config.RegistryBuilder;
import org.apache.http.conn.socket.ConnectionSocketFactory;
import org.apache.http.conn.socket.PlainConnectionSocketFactory;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustSelfSignedStrategy;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.DefaultHttpRequestRetryHandler;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Map;

@Service
public class HttpClientPool implements InitializingBean {
    @Value("${fateflow.url}")
    public String fateUrl;
    Logger logger = LoggerFactory.getLogger(HttpClientPool.class);
    private PoolingHttpClientConnectionManager poolConnManager;
    private RequestConfig requestConfig;
    private CloseableHttpClient httpClient;

    private static void config(HttpRequestBase httpRequestBase) {
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(8000)
                .setConnectTimeout(8000)
                .setSocketTimeout(8000).build();
        httpRequestBase.addHeader("Content-Type", "application/json;charset=UTF-8");
        httpRequestBase.setConfig(requestConfig);
    }

    private void initPool() {
        try {
            SSLContextBuilder builder = new SSLContextBuilder();
            builder.loadTrustMaterial(null, new TrustSelfSignedStrategy());
            SSLConnectionSocketFactory sslsf = new SSLConnectionSocketFactory(builder.build());
            Registry<ConnectionSocketFactory> socketFactoryRegistry = RegistryBuilder.<ConnectionSocketFactory>create().register(
                    "http", PlainConnectionSocketFactory.getSocketFactory()).register(
                    "https", sslsf).build();
            poolConnManager = new PoolingHttpClientConnectionManager(
                    socketFactoryRegistry);
            poolConnManager.setMaxTotal(200);
            poolConnManager.setDefaultMaxPerRoute(200);

            int socketTimeout = 10000;
            int connectTimeout = 10000;
            int connectionRequestTimeout = 10000;
            requestConfig = RequestConfig.custom().setConnectionRequestTimeout(
                    connectionRequestTimeout).setSocketTimeout(socketTimeout).setConnectTimeout(
                    connectTimeout).build();
            httpClient = getConnection();
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
            e.printStackTrace();
        }
    }

    private CloseableHttpClient getConnection() {
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(poolConnManager)
                .setDefaultRequestConfig(requestConfig)

                .setRetryHandler(new DefaultHttpRequestRetryHandler(0, false))
                .build();
        return httpClient;
    }


    public String post(String url, Map<String, Object> requestData) {
        HttpPost httpPost = new HttpPost(url);
        config(httpPost);
        StringEntity stringEntity = new StringEntity(JSON.toJSONString(requestData), "UTF-8");
        stringEntity.setContentEncoding(Charsets.UTF_8.toString());
        httpPost.setEntity(stringEntity);
        return getResponse(httpPost);
    }


    public String post(String url, String requestData) {

        HttpPost httpPost = new HttpPost(url);
        config(httpPost);
        StringEntity stringEntity = new StringEntity(requestData, Charsets.UTF_8.toString());
        stringEntity.setContentEncoding("UTF-8");
        httpPost.setEntity(stringEntity);
        String result = getResponse(httpPost);
        logger.info("httpclient sent url {} request {} result: {}", url, requestData, result);
        return result;
    }

    public String get(String url) {
        HttpGet httpGet = new HttpGet(url);
        config(httpGet);
        return getResponse(httpGet);
    }

    private String getResponse(HttpRequestBase request) {
        CloseableHttpResponse response = null;
        try {

            response = httpClient.execute(request,
                    HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            String result = EntityUtils.toString(entity, Charsets.UTF_8);
            EntityUtils.consume(entity);

            return result;
        } catch (IOException e) {
            logger.error("send http error",e);
            return "";
        } finally {
            try {
                if (response != null) {
                    response.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }


        }
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        initPool();
    }
}
