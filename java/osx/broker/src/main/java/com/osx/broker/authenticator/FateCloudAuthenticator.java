//package com.osx.broker.authenticator;
//
//
//import com.osx.core.config.MetaInfo;
//import com.webank.ai.eggroll.api.networking.proxy.Proxy;
//import org.apache.commons.lang3.StringUtils;
//import org.apache.commons.lang3.reflect.MethodUtils;
//import org.json.JSONObject;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import javax.security.sasl.AuthenticationException;
//import java.lang.reflect.Method;
//import java.net.ConnectException;
//import java.util.HashMap;
//import java.util.UUID;
//
//public class FateCloudAuthenticator implements Authenticator {
//    Logger logger = LoggerFactory.getLogger(FateCloudAuthenticator.class);
//
//    Method getSecretInfoMethod, signMethod, authenticateMethod;
//    Object fateCloud;
//
//    public FateCloudAuthenticator() throws Exception {
//        Class<?> clazz = Class.forName("com.webank.ai.fate.cloud.sdk.sdk.Fatecloud");
//        this.fateCloud = clazz.newInstance();
//
//        this.getSecretInfoMethod = MethodUtils.getMatchingMethod(clazz, "getSecretInfo", String.class, String.class);
//        this.signMethod = MethodUtils.getMatchingAccessibleMethod(clazz, "generateSignature",
//                String.class, String.class, String.class, String.class, String.class, String.class, String.class, String.class);
//        this.authenticateMethod = MethodUtils.getMatchingMethod(clazz, "checkPartyId", String.class, HashMap.class, String.class);
//
//    }
//
//    @Override
//    public JSONObject getSecretInfo() throws Exception {
//        // generate signature
//        int myPartyId = MetaInfo.ROLLSITE_PARTY_ID;
//        String secretInfoUrl = MetaInfo.TRANSFER_FATECLOUD_SECRET_INFO_URL;
//        String appSecret;
//        String appKey;
//        String role;
//        if (MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_USE_CONFIG) {
//            if (logger.isDebugEnabled()) {
//                logger.debug("manual configuration enabled, getting appKey, appSecret and party role from eggroll.properties");
//            }
//            appKey = MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_APPKEY;
//            appSecret = MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_APPSERCRET;
//            if (appKey == null || appSecret == null) {
//                throw new IllegalArgumentException("failed to get appKey or appSecret or party role from eggroll.properties");
//            }
//            switch (MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_ROLE.toLowerCase()) {
//                case "guest":
//                    role = "1";
//                case "host":
//                    role = "2";
//                default:
//                    throw new AuthenticationException("unsupported role：{}" + MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_ROLE);
//            }
//
//        } else {
//            String args = secretInfoUrl + "," + myPartyId;
//            logger.info("getSecretInfo of fateCloud calling, args:{}", args);
//
//            Object result = getSecretInfoMethod.invoke(fateCloud, secretInfoUrl, String.valueOf(myPartyId));
//
//            logger.info("getSecretInfo of fateCloud called");
//            if (result == null || StringUtils.isBlank(result.toString())) {
//                throw new AuthenticationException("result of getSecretInfo is empty");
//            }
//
//            JSONObject secretInfo = new JSONObject(result.toString());
//            if (secretInfo.get("data") == JSONObject.NULL) {
//                logger.error("partyID:${myPartyId} not registered");
//                throw new AuthenticationException("partyID= " + myPartyId + " is not registered");
//            }
//
//            appSecret = secretInfo.getJSONObject("data").getString("appSecret");
//            appKey = secretInfo.getJSONObject("data").getString("appKey");
//            if (logger.isDebugEnabled()) {
//                logger.debug("role of {} is {}", myPartyId, secretInfo.getJSONObject("data").getString("role"));
//            }
//            role = "guest".equals(secretInfo.getJSONObject("data").getString("role").toLowerCase()) ? "1" : "2";
//        }
//        JSONObject secretInfo = new JSONObject();
//        secretInfo.put("appSecret", appSecret);
//        secretInfo.put("appKey", appKey);
//        secretInfo.put("role", role);
//        return secretInfo;
//    }
//
//    @Override
//    public String sign() throws Exception {
//
//        JSONObject secretInfoGen = getSecretInfo();
//        String appSecret = (String) secretInfoGen.get("appSecret");
//        String appKey = (String) secretInfoGen.get("appKey");
//        String role = (String) secretInfoGen.get("role");
//        String myPartyId = String.valueOf(MetaInfo.ROLLSITE_PARTY_ID);
//        String time = String.valueOf(System.currentTimeMillis());
//        String uuid = UUID.randomUUID().toString();
//        String nonce = uuid.replaceAll("-", "");
//        String httpURI = MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_URI;
//        String body = "";
//        if (logger.isDebugEnabled()) {
//            logger.debug("generateSignature of fateCloud calling");
//        }
//        Object signature = signMethod.invoke(fateCloud, appSecret, myPartyId, role, appKey, time, nonce, httpURI, body);
//        if (logger.isDebugEnabled()) {
//            logger.debug("generateSignature of fateCloud called, signature={}", signature);
//        }
//
//        if (signature == null) {
//            throw new AuthenticationException("signature generated is null, please check args appSecret=" + appSecret + " , partyId=" + myPartyId + " , " +
//                    "role=" + role + " , appKey=" + appKey + " , time=" + time + " , nonce=" + nonce + " , httpURI=" + httpURI + " , body=" + body);
//        }
//
//        JSONObject authInfo = new JSONObject();
//        authInfo.put("signature", signature.toString());
//        authInfo.put("appKey", appKey);
//        authInfo.put("timestamp", time);
//        authInfo.put("nonce", nonce);
//        authInfo.put("role", role);
//        authInfo.put("httpUri", httpURI);
//
//        if (logger.isDebugEnabled()) {
//            logger.debug("authInfo to be sent " + authInfo);
//        }
//
//        return authInfo.toString();
//    }
//
//    @Override
//    public Boolean authenticate(Proxy.PollingFrame req) throws Exception {
//
//        String authUrl = MetaInfo.TRANSFER_FATECLOUD_AUTHENTICATION_URL;
//        String authString = req.getMetadata().getTask().getModel().getDataKey();
//        if (logger.isDebugEnabled()) {
//            logger.debug("req metaData recv={}", req.getMetadata().toString());
//        }
//        if ("".equals(authString)) {
//            throw new AuthenticationException("failed to get authentication info from header=" + req.getMetadata().toString());
//        }
//        JSONObject authInfo = new JSONObject(authString);
//        HashMap<String, String> heads = new HashMap<>();
//        heads.put("TIMESTAMP", authInfo.getString("timestamp"));
//        heads.put("PARTY_ID", req.getMetadata().getDst().getPartyId());
//        heads.put("NONCE", authInfo.getString("nonce"));
//        heads.put("ROLE", authInfo.getString("role"));
//        heads.put("APP_KEY", authInfo.getString("appKey"));
//        heads.put("URI", authInfo.getString("httpUri"));
//        heads.put("SIGNATURE", authInfo.getString("signature"));
//        String body = "";
//        if (logger.isDebugEnabled()) {
//            logger.debug("auth heads:" + heads.toString());
//            logger.debug("checkPartyId of fateCloud calling");
//        }
//
//        try {
//            Boolean result = (Boolean) authenticateMethod.invoke(fateCloud, authUrl, heads, body);
//            if (logger.isDebugEnabled()) {
//                logger.debug("checkPartyId of fateCloud called");
//            }
//            return result;
//        } catch (Exception e) {
//            if (e instanceof ConnectException) {
//                logger.error("server authenticate failed to connect to {}", authUrl);
//                throw new ConnectException(" server authenticate failed to connect to authentication server");
//            } else if (e instanceof AuthenticationException) {
//                throw new AuthenticationException("failed to authenticate, please check  client authentication info=" + authString, e);
//            } else {
//                throw new Exception("failed to authenticate, please check  client authentication info=" + authString, e);
//            }
//        }
//
//    }
//}
