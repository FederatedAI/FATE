package com.osx.broker.eggroll;

import com.webank.eggroll.core.meta.Meta;

public class ErEndpoint extends BaseProto<Meta.Endpoint> {

    String host;
    int port;

    public ErEndpoint(String ip, int port) {
        this.host = ip;
        this.port = port;
    }

    public static ErEndpoint parseFromPb(Meta.Endpoint endpoint) {
        if (endpoint != null) {
            ErEndpoint erEndpoint = new ErEndpoint(endpoint.getHost(), endpoint.getPort());
            return erEndpoint;
        }
        return null;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    @Override
    Meta.Endpoint toProto() {
        return Meta.Endpoint.newBuilder().setHost(host).setPort(port).build();

    }
}
