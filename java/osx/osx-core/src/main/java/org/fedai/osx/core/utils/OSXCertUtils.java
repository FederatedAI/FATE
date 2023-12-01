package org.fedai.osx.core.utils;


import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.router.RouterInfo;
import sun.misc.BASE64Decoder;
import sun.security.x509.X509CertImpl;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import java.io.*;
import java.security.*;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.concurrent.atomic.AtomicInteger;

/***
 * certificates type conversion
 */
public class OSXCertUtils {
    private static final int I0 = 0;
    private static final int I1 = 1;
    private static final String type = "PKCS12";
    private static final AtomicInteger keyStoreCount = new AtomicInteger(1);

    /***
     * x509 certificate packaged into p12 certificate
     * @param chain cert chain, issue cert +> superior cert +> ...
     * @param privateKey issued cert private key
     * @param filePath path to save p12 the cert
     * @param alias alias
     * @throws Exception NoCert,  NoSuchAlgorithm , NoKeyStore, io
     */
    public static void x509ToPkCS12(Certificate[] chain, Key privateKey, String filePath, String alias) throws Exception {
        try (OutputStream os = new FileOutputStream(filePath)) {
            KeyStore keyStore = KeyStore.getInstance(type);
            keyStore.load(null);
            keyStore.setKeyEntry(alias, privateKey, MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray(), chain);
            keyStore.store(os, MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray());
        }
    }


    /***
     * get x509 cert and private key by p12 cert
     * @param filePath p12 cert file
     * @param cs  p1:storePassword  p2:certPassword
     * @return x509 certificate and private key
     * @throws Exception NoCert,  NoSuchAlgorithm , NoKeyStore, io
     */
    public static X509AndKey getX509AndKeyByPkCS12(String filePath, String... cs) throws Exception {
        try (InputStream is = new FileInputStream(filePath)) {
            KeyStore keyStore = KeyStore.getInstance(type);
            keyStore.load(is, toCharArray(I0, cs));
            String alias = keyStore.aliases().nextElement();
            return new X509AndKey(((X509CertImpl) keyStore.getCertificate(alias)), ((PrivateKey) keyStore.getKey(alias, toCharArray(I1, cs))));
        }
    }

    public static SSLContext getSSLContext(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        KeyStore keyStore = getKeyStore(caPath, clientCertPath, clientKeyPath);
        // Initialize the ssl context object
        SSLContext sslContext = SSLContext.getInstance("SSL");
        TrustManager[] tm = {OsxX509TrustManager.getInstance(keyStore)};
        // Load client certificate
        KeyManagerFactory kmf = KeyManagerFactory.getInstance("SunX509");
        kmf.init(keyStore, MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray());
        sslContext.init(kmf.getKeyManagers(), tm, new SecureRandom());
        return sslContext;
    }

    public static SSLContext getSSLContext(RouterInfo routerInfo) throws Exception {
        SSLContext sslContext = SSLContext.getInstance("TLS");
        if (routerInfo.isUseKeyStore()) {
            KeyStore keyStore = loadKeyStore(routerInfo.getKeyStoreFilePath(), routerInfo.getKeyStorePassword());
            KeyStore trustStore = loadKeyStore(routerInfo.getTrustStoreFilePath(), routerInfo.getTrustStorePassword());
            // 创建 KeyManagerFactory 和 TrustManagerFactory
            KeyManagerFactory keyManagerFactory = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            keyManagerFactory.init(keyStore, routerInfo.getKeyStorePassword().toCharArray());
            TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            trustManagerFactory.init(trustStore);

            // 初始化 SSLContext
            sslContext.init(keyManagerFactory.getKeyManagers(), trustManagerFactory.getTrustManagers(), null);

        } else {
            KeyStore keyStore = getKeyStore(routerInfo.getCaFile(), routerInfo.getCertChainFile(), routerInfo.getPrivateKeyFile());
            // Initialize the ssl context object
            TrustManager[] tm = {OsxX509TrustManager.getInstance(keyStore)};
            // Load client certificate
            KeyManagerFactory kmf = KeyManagerFactory.getInstance("SunX509");
            kmf.init(keyStore, MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray());
            sslContext.init(kmf.getKeyManagers(), tm, new SecureRandom());
            return sslContext;
        }
        return sslContext;
    }

    private static KeyStore loadKeyStore(String keyStorePath, String keyStorePassword) throws Exception {
        try (FileInputStream fis = new FileInputStream(keyStorePath)) {
            KeyStore keyStore = KeyStore.getInstance("JKS");
            keyStore.load(fis, keyStorePassword.toCharArray());
            return keyStore;
        }
    }

    public static KeyStore getKeyStore(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        KeyStore keyStore = KeyStore.getInstance("PKCS12");
        keyStore.load(null);
        keyStore.setKeyEntry(MetaInfo.PROPERTY_HTTP_SSL_CLIENT_KEY_STORE_ALIAS, importPrivateKey(clientKeyPath), MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray(), new Certificate[]{importCert(clientCertPath), importCert(caPath)});
        return keyStore;
    }

    public static KeyStore getKeyStore2(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        KeyStore keyStore = KeyStore.getInstance("PKCS12");
        keyStore.load(null);
        keyStore.setKeyEntry(MetaInfo.PROPERTY_HTTP_SSL_SERVER_KEY_STORE_ALIAS, importPrivateKey(clientKeyPath), MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray(), new Certificate[]{importCert(clientCertPath), importCert(caPath)});
        return keyStore;
    }

    public static KeyStore getTrustStore(String keyStorePath, String trustStoreType) throws Exception {
        KeyStore keyStore = KeyStore.getInstance(trustStoreType.toUpperCase());
        keyStore.load(new FileInputStream(new File(keyStorePath)), MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray());
        return keyStore;
    }

  /*  public static String createKeyStore(String caPath, String clientCertPath, String clientKeyPath) throws Exception {
        PrivateKey privateKey = importPrivateKey(clientKeyPath);
//        Certificate[] certificates = {importCert(clientCertPath), importCert(caPath)};
        Certificate[] certificates = {importCert(clientCertPath), importCert(caPath)};
        String pfxPath = OSXCertUtils.getTempStorePath();
        File pfxFile = new File(pfxPath);
        FileUtils.createNewFile(pfxFile);
        OSXCertUtils.x509ToPkCS12(certificates, privateKey, pfxPath, MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_ALIAS);
        return pfxPath;
    }*/

    public static Certificate importCert(String certFile) throws Exception {
        try (FileInputStream certStream = new FileInputStream(certFile)) {
            CertificateFactory cf = CertificateFactory.getInstance("X.509");
            return cf.generateCertificate(certStream);
        }
    }

    // Import private key
    public static PrivateKey importPrivateKey(String privateKeyFile) throws Exception {
       /* try (FileInputStream keyStream = new FileInputStream(privateKeyFile)) {
            String space = "";
            byte[] bytes = new byte[keyStream.available()];
            int length = keyStream.read(bytes);
            String keyString = new String(bytes, 0, length);
            if (keyString.startsWith("-----BEGIN PRIVATE KEY-----\n")) {
                keyString = keyString.replace("-----BEGIN PRIVATE KEY-----\n", space).replace("-----END PRIVATE KEY-----", space);
            }
            PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(new BASE64Decoder().decodeBuffer(keyString));
            return KeyFactory.getInstance("RSA").generatePrivate(keySpec);
        }*/
        String privateKey = readFileContent(privateKeyFile);
        PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(new BASE64Decoder().decodeBuffer(privateKey));
        return KeyFactory.getInstance("RSA").generatePrivate(keySpec);
    }

    private static String readFileContent(String filePath) throws Exception {
        File file = new File(filePath);
        StringBuffer key = new StringBuffer();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {

            String tempString = null;

            while ((tempString = reader.readLine()) != null) {
                if (!tempString.startsWith("--")) {
                    key.append(tempString);
                }
            }
        }
        return key.toString();
    }


    //determine whether the string is null and get the default string character array
    private static char[] toCharArray(int index, String... str) {
        return str.length <= index || str[index] == null ? MetaInfo.PROPERTY_HTTPS_SERVER_KEYSTORE_FILE_PASSWORD.toCharArray() : str[index].toCharArray();
    }

    public static String getTempStorePath() {
        return "";
    }

    /***
     * this class pack X509Certificate and privateKey
     */
    public static class X509AndKey {
        private final X509CertImpl x509Cert;
        private final PrivateKey privateKey;

        public X509AndKey(X509CertImpl x509Certificate, PrivateKey privateKey) {
            this.x509Cert = x509Certificate;
            this.privateKey = privateKey;
        }
    }
}
