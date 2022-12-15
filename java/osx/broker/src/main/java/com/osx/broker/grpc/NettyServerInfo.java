//package com.osx.transfer.grpc;
//
//import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
//
//public class NettyServerInfo {
//
//    public NettyServerInfo() {
//        this.negotiationType = NegotiationType.PLAINTEXT;
//    }
//
//    public NettyServerInfo(String negotiationType, String certChainFilePath, String privateKeyFilePath, String trustCertCollectionFilePath) {
//        this.negotiationType = NegotiationType.valueOf(negotiationType);
//        this.certChainFilePath = certChainFilePath;
//        this.privateKeyFilePath = privateKeyFilePath;
//        this.trustCertCollectionFilePath = trustCertCollectionFilePath;
//    }
//
//    private NegotiationType negotiationType;
//
//    private String certChainFilePath;
//
//    private String privateKeyFilePath;
//
//    private String trustCertCollectionFilePath;
//
//    public NegotiationType getNegotiationType() {
//        return negotiationType;
//    }
//
//    public void setNegotiationType(NegotiationType negotiationType) {
//        this.negotiationType = negotiationType;
//    }
//
//    public String getCertChainFilePath() {
//        return certChainFilePath;
//    }
//
//    public void setCertChainFilePath(String certChainFilePath) {
//        this.certChainFilePath = certChainFilePath;
//    }
//
//    public String getPrivateKeyFilePath() {
//        return privateKeyFilePath;
//    }
//
//    public void setPrivateKeyFilePath(String privateKeyFilePath) {
//        this.privateKeyFilePath = privateKeyFilePath;
//    }
//
//    public String getTrustCertCollectionFilePath() {
//        return trustCertCollectionFilePath;
//    }
//
//    public void setTrustCertCollectionFilePath(String trustCertCollectionFilePath) {
//        this.trustCertCollectionFilePath = trustCertCollectionFilePath;
//    }
//}
