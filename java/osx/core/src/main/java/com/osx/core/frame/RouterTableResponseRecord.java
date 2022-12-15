//package com.osx.core.frame;
//
//import java.util.List;
//
///**
// * @auther wangcaoyuan
// * @date 2021/10/21
// * @remark
// */
//public class RouterTableResponseRecord {
//    String partyId;
//    Long createTime;
//    Long updateTime;
//    List<RouterTable> routerList;
//    int count;
//
//    public static class RouterTable {
//        String ip;
//        Integer port;
//        boolean useSSL;
//        String negotiationType;
//        String certChainFile;
//        String privateKeyFile;
//        String caFile;
//        String serverType;
//        public String getIp() {
//            return ip;
//        }
//
//        public void setIp(String ip) {
//            this.ip = ip;
//        }
//
//        public Integer getPort() {
//            return port;
//        }
//
//        public void setPort(Integer port) {
//            this.port = port;
//        }
//
//        public boolean isUseSSL() {
//            return useSSL;
//        }
//
//        public void setUseSSL(boolean useSSL) {
//            this.useSSL = useSSL;
//        }
//
//        public String getNegotiationType() {
//            return negotiationType;
//        }
//
//        public void setNegotiationType(String negotiationType) {
//            this.negotiationType = negotiationType;
//        }
//
//        public String getCertChainFile() {
//            return certChainFile;
//        }
//
//        public void setCertChainFile(String certChainFile) {
//            this.certChainFile = certChainFile;
//        }
//
//        public String getPrivateKeyFile() {
//            return privateKeyFile;
//        }
//
//        public void setPrivateKeyFile(String privateKeyFile) {
//            this.privateKeyFile = privateKeyFile;
//        }
//
//        public String getCaFile() {
//            return caFile;
//        }
//
//        public void setCaFile(String caFile) {
//            this.caFile = caFile;
//        }
//
//        public String getServerType() {
//            return serverType;
//        }
//
//        public void setServerType(String serverType) {
//            this.serverType = serverType;
//        }
//    }
//
//    public String getPartyId() {
//        return partyId;
//    }
//
//    public void setPartyId(String partyId) {
//        this.partyId = partyId;
//    }
//
//    public Long getCreateTime() {
//        return createTime;
//    }
//
//    public void setCreateTime(Long createTime) {
//        this.createTime = createTime;
//    }
//
//    public Long getUpdateTime() {
//        return updateTime;
//    }
//
//    public void setUpdateTime(Long updateTime) {
//        this.updateTime = updateTime;
//    }
//
//    public int getCount() {
//        return count;
//    }
//
//    public void setCount(int count) {
//        this.count = count;
//    }
//
//    public List<RouterTable> getRouterList() {
//        return routerList;
//    }
//
//    public void setRouterList(List<RouterTable> routerList) {
//        this.routerList = routerList;
//    }
//}
