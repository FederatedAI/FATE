package com.osx.broker.zk;

import com.google.common.collect.Lists;

import java.util.List;

public class ZkConfig {



    public ZkConfig(String address,int timeout){
        this.address = address;
        this.timeout = timeout;
    }
    String address;
    int  timeout;
    public int getTimeout() {
        return timeout;
    }
    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }
    public String getAddress() {
        return address;
    }
    public void setAddress(String address) {
        this.address = address;
    }
    public List<String> toIpPortList(){
       return Lists.newArrayList(address.split(","));
    }
}
