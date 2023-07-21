package com.osx.broker.http;

import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.utils.OSXCertUtils;
import com.osx.core.utils.OsxX509TrustManager;
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

import javax.net.ssl.*;
import java.io.IOException;
import java.security.*;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class HttpsClientPool {
    private static final Logger logger = LoggerFactory.getLogger(HttpsClientPool.class);
    private static final Map<String, CloseableHttpClient> httpsClientPool = new HashMap<>();

    public static CloseableHttpClient getConnection(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        String certKey = buildCertKey(caPath, clientCertPath, clientKeyPath);
        CloseableHttpClient httpClient = httpsClientPool.get(certKey);
        if (httpClient == null) {
            httpClient = createConnection(caPath, clientCertPath, clientKeyPath);
            httpsClientPool.put(certKey, httpClient);
        }
        return httpClient;
    }

    private static String buildCertKey(String caPath, String clientCertPath, String clientKeyPath) {
        return caPath + "_" + clientCertPath + "_" + clientKeyPath;
    }

    public static CloseableHttpClient createConnection(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT)
                .setConnectTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_CONN_TIME_OUT)
                .setSocketTimeout(MetaInfo.PROPERTY_HTTP_CLIENT_CONFIG_SOCK_TIME_OUT).build();
        CloseableHttpClient httpClient = null;
        try {
            SSLContextBuilder builder = new SSLContextBuilder();
            builder.loadTrustMaterial(null, new TrustSelfSignedStrategy());
            SSLConnectionSocketFactory sslsf;
            if (MetaInfo.PROPERTY_HTTP_SSL_HOSTNAME_VERIFY) {
                sslsf = new SSLConnectionSocketFactory(OSXCertUtils.getSSLContext(caPath, clientCertPath, clientKeyPath));
            } else {
                sslsf = new SSLConnectionSocketFactory(OSXCertUtils.getSSLContext(caPath, clientCertPath, clientKeyPath), OsxX509TrustManager.HostnameVerifier2.getInstance());
            }
            Registry<ConnectionSocketFactory> socketFactoryRegistry = RegistryBuilder.<ConnectionSocketFactory>create().register(
                    Dict.HTTP, PlainConnectionSocketFactory.getSocketFactory()).register(
                    Dict.HTTPS, sslsf).build();
            PoolingHttpClientConnectionManager poolConnManager = new PoolingHttpClientConnectionManager(
                    socketFactoryRegistry);
            poolConnManager.setMaxTotal(MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_MAX_TOTAL);
            poolConnManager.setDefaultMaxPerRoute(MetaInfo.PROPERTY_HTTP_CLIENT_INIT_POOL_DEF_MAX_PER_ROUTE);
            httpClient = HttpClients.custom()
                    .setSSLSocketFactory(sslsf)
                    .setConnectionManager(poolConnManager)
                    .setDefaultRequestConfig(requestConfig)
                    .evictExpiredConnections()
                    .evictIdleConnections(MetaInfo.PROPERTY_HTTP_CLIENT_MAX_IDLE_TIME, TimeUnit.SECONDS)
                    .setRetryHandler(new DefaultHttpRequestRetryHandler(0, false))
                    .build();
        } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException ex) {
            logger.error("init https client pool failed:", ex);
        }
        return httpClient;
    }

    public static Osx.Outbound sendPtpPost(String url, byte[] body, Map<String, String> headers, String caPath, String clientCertPath, String clientKeyPath) throws Exception {

        HttpPost httpPost = new HttpPost(url);
        HttpClientPool.config(httpPost, headers);
        if (body != null) {
            ByteArrayEntity byteArrayEntity = new ByteArrayEntity(body);
            httpPost.setEntity(byteArrayEntity);
        }
        return getPtpHttpsResponse(httpPost, caPath, clientCertPath, clientKeyPath);
    }

    @SuppressWarnings("unused")
    public static String sendPost(String url, byte[] body, Map<String, String> headers, String caPath, String clientCertPath, String clientKeyPath) {
        HttpPost httpPost = new HttpPost(url);
        HttpClientPool.config(httpPost, headers);
        ByteArrayEntity byteArrayEntity = new ByteArrayEntity(body);
        httpPost.setEntity(byteArrayEntity);
        return getResponse(httpPost, caPath, clientCertPath, clientKeyPath);
    }

    public static String get(String url, Map<String, String> headers, String caPath, String clientCertPath, String clientKeyPath) {
        return sendGet(url, headers, caPath, clientCertPath, clientKeyPath);
    }

    public static String get(String url, String caPath, String clientCertPath, String clientKeyPath) {
        return sendGet(url, null, caPath, clientCertPath, clientKeyPath);
    }

    public static String sendGet(String url, Map<String, String> headers, String caPath, String clientCertPath, String clientKeyPath) {
        HttpGet httpGet = new HttpGet(url);
        HttpClientPool.config(httpGet, headers);
        return getResponse(httpGet, caPath, clientCertPath, clientKeyPath);
    }

    private static String getResponse(HttpRequestBase request, String caPath, String clientCertPath, String clientKeyPath) {
        CloseableHttpResponse response = null;
        try {
            response = getConnection(caPath, clientCertPath, clientKeyPath).execute(request, HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            String result = EntityUtils.toString(entity, Dict.CHARSET_UTF8);
            EntityUtils.consume(entity);
            return result;
        } catch (Exception ex) {
            logger.error("get https response failed:", ex);
            return null;
        } finally {
            try {
                if (response != null) {
                    response.close();
                }
            } catch (IOException ex) {
                logger.error("get https response failed:", ex);
            }
        }
    }

    private static Osx.Outbound getPtpHttpsResponse(HttpRequestBase request, String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
        CloseableHttpResponse response = null;
        try {
            response = getConnection(caPath, clientCertPath, clientKeyPath).execute(request, HttpClientContext.create());
            HttpEntity entity = response.getEntity();
            byte[] payload = EntityUtils.toByteArray(entity);
            Header[] headers = response.getAllHeaders();
            Map<String, String> headMap = Maps.newHashMap();
            if (headers != null) {
                for (Header temp : headers) {
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
            logger.error("get https response failed:", ex);
            ex.printStackTrace();
            throw  ex;
        } finally {
            try {
                if (response != null) {
                    response.close();
                }
            } catch (IOException ex) {
                logger.error("get https response failed:", ex);
            }
        }
    }

    @SuppressWarnings("unused")
    private static SSLSocketFactory getSslFactory(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        KeyStore keyStore = OSXCertUtils.getKeyStore(caPath, clientCertPath, clientKeyPath);
        // Initialize the ssl context object
        SSLContext sslContext = SSLContext.getInstance("SSL");
        TrustManager[] tm = {OsxX509TrustManager.getInstance(keyStore)};
        // Load client certificate
        KeyManagerFactory kmf = KeyManagerFactory.getInstance("SunX509");
        kmf.init(keyStore, MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_PASSWORD.toCharArray());
        sslContext.init(kmf.getKeyManagers(), tm, new SecureRandom());
        // Initialize the factory
        return sslContext.getSocketFactory();
    }


}
