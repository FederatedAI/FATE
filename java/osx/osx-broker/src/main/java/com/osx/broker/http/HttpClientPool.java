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

package com.osx.broker.http;

import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.PtpHttpHeader;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.http.Header;
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
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.DefaultHttpRequestRetryHandler;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class HttpClientPool {
    private static final Logger logger = LoggerFactory.getLogger(HttpClientPool.class);
    private static PoolingHttpClientConnectionManager poolConnManager;
    private static CloseableHttpClient httpClient;

    static void config(HttpRequestBase httpRequestBase, Map<String, String> headers) {
        Integer reqTimeout = null;
        Integer connectionTimeout = null;
        Integer socketTimeout = null;

        if (MetaInfo.PROPERTY_HTTP_CLIENT_METHOD_CONFIG_MAP != null) {
            Map<String, Integer> methodConfig = MetaInfo.PROPERTY_HTTP_CLIENT_METHOD_CONFIG_MAP.get(headers.get(PtpHttpHeader.SourceMethod));
            if (methodConfig != null) {
                reqTimeout = methodConfig.get(Dict.METHOD_CONFIG_REQ_TIMEOUT);
                connectionTimeout = methodConfig.get(Dict.METHOD_CONFIG_CONNECTION_TIMEOUT);
                socketTimeout = methodConfig.get(Dict.METHOD_CONFIG_SOCKET_TIMEOUT);

            }
        }

        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(ObjectUtils.firstNonNull(reqTimeout, MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT))
                .setConnectTimeout(ObjectUtils.firstNonNull(connectionTimeout, MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_TIME_OUT))
                .setSocketTimeout(ObjectUtils.firstNonNull(socketTimeout, MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_SOCK_TIME_OUT)).build();
        httpRequestBase.addHeader(Dict.CONTENT_TYPE, Dict.CONTENT_TYPE_JSON_UTF8);
        if (headers != null) {
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
                    Dict.HTTP, PlainConnectionSocketFactory.getSocketFactory()).register(
                    Dict.HTTPS, sslsf).build();
            poolConnManager = new PoolingHttpClientConnectionManager(
                    socketFactoryRegistry);
            poolConnManager.setMaxTotal(MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_MAX_TOTAL);
            poolConnManager.setDefaultMaxPerRoute(MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_DEF_MAX_PER_ROUTE);
            httpClient = createConnection();
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException ex) {
            logger.error("init http client pool failed:", ex);
        }
    }

    public static CloseableHttpClient getConnection() {
        return httpClient;
    }

    public static CloseableHttpClient createConnection() {
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT)
                .setConnectTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_TIME_OUT)
                .setSocketTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_SOCK_TIME_OUT).build();
        CloseableHttpClient httpClient = HttpClients.custom()
                .setConnectionManager(poolConnManager)
                .setDefaultRequestConfig(requestConfig)
                .evictExpiredConnections()
                .evictIdleConnections(5, TimeUnit.SECONDS)
                .setRetryHandler(new DefaultHttpRequestRetryHandler(0, false))
                .build();
        return httpClient;
    }

    public static Osx.Outbound sendPtpPost(String url, byte[] body, Map<String, String> headers) {

        HttpPost httpPost = new HttpPost(url);
        config(httpPost, headers);
        if (body != null) {
            ByteArrayEntity byteArrayEntity = new ByteArrayEntity(body);
            httpPost.setEntity(byteArrayEntity);
        }
        return getPtpHttpResponse(httpPost);
    }

    public static String sendPost(String url, byte[] body, Map<String, String> headers) {
        HttpPost httpPost = new HttpPost(url);
        config(httpPost, headers);
        ByteArrayEntity byteArrayEntity = new ByteArrayEntity(body);
        httpPost.setEntity(byteArrayEntity);
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
            response = httpClient.execute(request, HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            String result = EntityUtils.toString(entity, Dict.CHARSET_UTF8);
            EntityUtils.consume(entity);
            return result;
        } catch (IOException ex) {
            logger.error("get http response failed:", ex);
            return null;
        } finally {
            try {
                if (response != null) {
                    response.close();
                }
            } catch (IOException ex) {
                logger.error("get http response failed:", ex);
            }
        }
    }

    private static Osx.Outbound getPtpHttpResponse(HttpRequestBase request) {

        Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
        CloseableHttpResponse response = null;
        try {
            response = httpClient.execute(request, HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            byte[] payload = EntityUtils.toByteArray(entity);
            Header[] headers = response.getAllHeaders();
            Map<String, String> headMap = Maps.newHashMap();
            if (headers != null) {
                for (int i = 0; i < headers.length; i++) {
                    Header temp = headers[i];
                    headMap.put(temp.getName(), temp.getValue());
                }
            }
            if (payload != null)
                outboundBuilder.setPayload(ByteString.copyFrom(payload));
            if (headMap.get(PtpHttpHeader.ReturnCode) != null)
                outboundBuilder.setCode(headMap.get(PtpHttpHeader.ReturnCode));
            if (headMap.get(PtpHttpHeader.ReturnMessage) != null)
                outboundBuilder.setMessage(headMap.get(PtpHttpHeader.ReturnMessage));

            EntityUtils.consume(entity);
            return outboundBuilder.build();
        } catch (IOException ex) {
            logger.error("get http response failed:", ex);
            ex.printStackTrace();
            return null;
        } finally {
            try {
                if (response != null) {
                    response.close();
                }
            } catch (IOException ex) {
                logger.error("get http response failed:", ex);
            }
        }
    }

//    public static String transferPost(String url, Map<String, Object> requestData) {
//        HttpPost httpPost = new HttpPost(url);
//        RequestConfig requestConfig = RequestConfig.custom()
//                .setConnectionRequestTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT)
//                .setConnectTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_TIME_OUT)
//                .setSocketTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_SOCK_TIME_OUT).build();
//        httpPost.addHeader(Dict.CONTENT_TYPE, Dict.CONTENT_TYPE_JSON_UTF8);
//        httpPost.setConfig(requestConfig);
//        StringEntity stringEntity = new StringEntity(JsonUtil.object2Json(requestData), Dict.CHARSET_UTF8);
//        stringEntity.setContentEncoding(Dict.CHARSET_UTF8);
//        httpPost.setEntity(stringEntity);
//        return getResponse(httpPost);
//    }
}
