package com.osx.broker.service;

import com.osx.broker.zk.CuratorZookeeperClient;
import com.osx.broker.zk.ZkConfig;

public class RegisterService {


    public RegisterService(ZkConfig  zkConfig){
        curatorZookeeperClient = new CuratorZookeeperClient(zkConfig);
    }
    CuratorZookeeperClient curatorZookeeperClient ;



    public  void  registerTransferQueue(){

    }
    public  void registerComponent(){

    }


}
