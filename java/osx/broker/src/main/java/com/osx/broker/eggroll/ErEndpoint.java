package com.osx.broker.eggroll;

import com.webank.eggroll.core.meta.Meta;

public class ErEndpoint extends BaseProto<Meta.Endpoint>{

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public ErEndpoint(String ip, int port){
        this.host=ip;
        this.port= port;
    }
    String host;

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    int port;


    public  static ErEndpoint  parseFromPb(Meta.Endpoint  endpoint){
        if(endpoint!=null){
            ErEndpoint  erEndpoint = new  ErEndpoint(endpoint.getHost(), endpoint.getPort());
            return  erEndpoint;
        }
        return null;
    }

    @Override
    Meta.Endpoint toProto() {
       return  Meta.Endpoint.newBuilder().setHost(host).setPort(port).build();

    }
}
