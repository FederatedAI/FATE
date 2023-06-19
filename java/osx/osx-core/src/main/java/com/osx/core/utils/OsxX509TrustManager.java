package com.osx.core.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.*;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;

public class OsxX509TrustManager implements X509TrustManager {
    private static final Logger logger = LoggerFactory.getLogger(OsxX509TrustManager.class);
    public static final String tabs = "%2F", equalSign = "%3D";

    private final X509TrustManager x509TrustManager;

    public OsxX509TrustManager(X509TrustManager x509TrustManager) {
        this.x509TrustManager = x509TrustManager;
    }

    @Override
    public void checkClientTrusted(X509Certificate[] chain, String authType) {
        try {
            if (this.x509TrustManager == null) return;
            this.x509TrustManager.checkClientTrusted(chain, authType);
        } catch (CertificateException exc) {
            logger.error(exc.getMessage());
        }
    }

    @Override
    public void checkServerTrusted(X509Certificate[] chain, String authType) {
        // sunJSSEX509TrustManager.checkServerTrusted(chain, authType);
//        if (checkServer) {
//            for (X509Certificate x509Certificate : chain) {
//                // Use ca certificate verify
//                verify(caX509Certificate, x509Certificate);
//
//                // Send ocsp request verify
//                ocspVerify(x509Certificate);
//            }
//        }
    }

    @Override
    public X509Certificate[] getAcceptedIssuers() {
        if (this.x509TrustManager == null) return null;
        return this.x509TrustManager.getAcceptedIssuers();
    }

    public static OsxX509TrustManager getInstance() {
        return new OsxX509TrustManager(null);
    }

    public static OsxX509TrustManager getInstance(KeyStore keyStore) throws NoSuchProviderException, NoSuchAlgorithmException, KeyStoreException {
        X509TrustManager x509TrustManager = null;
        TrustManagerFactory tmf = TrustManagerFactory.getInstance("SunX509", "SunJSSE");
        tmf.init(keyStore);
        TrustManager[] tms = tmf.getTrustManagers();
        for (TrustManager tm : tms) {
            if (tm instanceof X509TrustManager) {
                x509TrustManager = (X509TrustManager) tm;
                break;
            }
        }
        return new OsxX509TrustManager(x509TrustManager);
    }

    // Verify that the certificate if expired, and is issued for the root certificate
//    public static void verify(X509Certificate superiorCert, X509Certificate issueCert) throws CertificateException {
//        try {
//            issueCert.checkValidity();
//            issueCert.verify(superiorCert.getPublicKey());
//        } catch (Exception e) {
//            throw new CertificateException(e);
//        }
//    }

    // Obtain ocsp service address from the certificate and verify the validity of the certificate
//    public static void ocspVerify(X509Certificate x509Certificate) throws CertificateException {
//        X509CertImpl x509Cert = (X509CertImpl) x509Certificate;
//        AuthorityInfoAccessExtension accessExtension = x509Cert.getAuthorityInfoAccessExtension();
//        List<AccessDescription> accessDescriptions = accessExtension.getAccessDescriptions();
//        for (AccessDescription accessDescription : accessDescriptions) {
//            String anObject = accessDescription.getAccessMethod().toString();
//            if ("ocsp".equals(anObject) || "1.3.6.1.5.5.7.48.1".equals(anObject)) {
//                GeneralName accessLocation = accessDescription.getAccessLocation();
//                URI ocspUrl = ((URIName) accessLocation.getName()).getURI();
//                goSendOCSP(ocspUrl.toString(), x509Cert);
//            }
//        }
//    }

    // Send ocsp request
//    public static void goSendOCSP(String ocspUrl, X509CertImpl x509Certificate) throws CertificateException {
//        try {
//            URL url = new URL(ocspUrl + "/" + getOcspRequestData(x509Certificate));
//            HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
//            urlConnection.setConnectTimeout(5000);
//            urlConnection.setReadTimeout(15000);
//            urlConnection.setRequestMethod("GET");
//            urlConnection.setDoOutput(true);
//            urlConnection.setDoInput(true);
//            urlConnection.setRequestProperty("Content-type", "application/ocsp-request");
//
//            try (InputStream br = urlConnection.getInputStream();
//                 ByteArrayOutputStream aos = new ByteArrayOutputStream()
//            ) {
//                int len;
//                byte[] bytes = new byte[br.available()];
//                while ((len = br.read(bytes)) != -1) {
//                    aos.write(bytes, 0, len);
//                }
//                OCSPResponse ocspResponse = new OCSPResponse(aos.toByteArray());
//                OCSPResponse.ResponseStatus responseStatus = ocspResponse.getResponseStatus();
//
//                if (!responseStatus.equals(OCSPResponse.ResponseStatus.SUCCESSFUL)) {
//                    throw new CertificateException("ocsp request failed, request state: " + responseStatus);
//                }
//
//                Set<CertId> certIds = ocspResponse.getCertIds();
//                for (CertId certId : certIds) {
//                    // Date nextUpdate = singleResponse.getNextUpdate();
//                    // CRLReason revocationReason = singleResponse.getRevocationReason();
//                    // Date thisUpdate = singleResponse.getThisUpdate();
//                    OCSPResponse.SingleResponse singleResponse = ocspResponse.getSingleResponse(certId);
//                    OCSP.RevocationStatus.CertStatus certStatus = singleResponse.getCertStatus();
//                    System.out.println("server certificate serial number: " + certId.getSerialNumber().toString(16) + ", status: " + certStatus);
//
//                    if (!OCSP.RevocationStatus.CertStatus.GOOD.equals(certStatus)) {
//                        // throw new CertificateException("服务器验证失败, 证书状态: " + certStatus);
//                    }
//                }
//
//
//            } catch (Exception e) {
//                throw new CertificateException(e);
//            }
//        } catch (IOException e) {
//            throw new CertificateException(e);
//        }
//    }

    // get ocsp request bytes
//    private static byte[] getOcspRequestBytesData(X509CertImpl x509Certificate) throws IOException {
//        return new OCSPRequest(new CertId(x509Certificate, x509Certificate.getSerialNumberObject())).encodeBytes();
//    }

    // get Base64 encode ocsp request url string parameter
//    private static String getOcspRequestData(X509CertImpl certificate) throws IOException {
//        CertId certId = new CertId(certificate, certificate.getSerialNumberObject());
//        OCSPRequest request = new OCSPRequest(certId);
//        String encodeBuffer = new BASE64Encoder().encodeBuffer(request.encodeBytes());
//        return encodeBuffer.replace("\r\n", "").replace("/", tabs).replace("=", equalSign);
//    }

    // OCSPRequest
//    private static class OCSPRequest {
//        private static final Debug debug = Debug.getInstance("certpath");
//        private static final boolean dump;
//        private final List<CertId> certIds;
//        private final List<java.security.cert.Extension> extensions;
//        private byte[] nonce;
//
//        public OCSPRequest(CertId certId) {
//            this(Collections.singletonList(certId));
//        }
//
//        public OCSPRequest(List<CertId> certIdList) {
//            this.certIds = certIdList;
//            this.extensions = Collections.emptyList();
//        }
//
//        public OCSPRequest(List<CertId> certIdList, List<java.security.cert.Extension> extensionList) {
//            this.certIds = certIdList;
//            this.extensions = extensionList;
//        }
//
//        public byte[] encodeBytes() throws IOException {
//            DerOutputStream fillDOS = new DerOutputStream();
//            DerOutputStream certIdDOS = new DerOutputStream();
//
//            for (CertId certId : this.certIds) {
//                DerOutputStream encodeDos = new DerOutputStream();
//                certId.encode(encodeDos);
//                certIdDOS.write((byte) 48, encodeDos);
//            }
//
//            fillDOS.write((byte) 48, certIdDOS);
//            DerOutputStream extensionDos;
//            DerOutputStream endDos;
//            if (!this.extensions.isEmpty()) {
//                extensionDos = new DerOutputStream();
//
//                for (java.security.cert.Extension extension : this.extensions) {
//                    extension.encode(extensionDos);
//                    if (extension.getId().equals(PKIXExtensions.OCSPNonce_Id.toString())) {
//                        this.nonce = extension.getValue();
//                    }
//                }
//
//                endDos = new DerOutputStream();
//                endDos.write((byte) 48, extensionDos);
//                fillDOS.write(DerValue.createTag((byte) -128, true, (byte) 2), endDos);
//            }
//
//            extensionDos = new DerOutputStream();
//            extensionDos.write((byte) 48, fillDOS);
//            endDos = new DerOutputStream();
//            endDos.write((byte) 48, extensionDos);
//            byte[] bytes = endDos.toByteArray();
//            if (dump) {
//                HexDumpEncoder dumpEncoder = new HexDumpEncoder();
//                debug.println("OCSPRequest bytes...\n\n" + dumpEncoder.encode(bytes) + "\n");
//            }
//
//            return bytes;
//        }
//
//        public List<CertId> getCertIds() {
//            return this.certIds;
//        }
//
//        public byte[] getNonce() {
//            return this.nonce;
//        }
//
//        static {
//            dump = debug != null && Debug.isOn("ocsp");
//        }
//    }

    public static class HostnameVerifier2 implements HostnameVerifier {

        @Override
        public boolean verify(String s, SSLSession sslSession) {
            return true;
        }

        public static HostnameVerifier2 getInstance() {
            return new HostnameVerifier2();
        }
    }
}
