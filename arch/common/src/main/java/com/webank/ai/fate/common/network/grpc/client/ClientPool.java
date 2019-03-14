package com.webank.ai.fate.common.network.grpc.client;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import com.webank.ai.fate.common.statuscode.ReturnCode;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ClientPool {
    private static final Logger LOGGER = LogManager.getLogger();
    private static ConcurrentHashMap<String, ManagedChannel> pool = new ConcurrentHashMap();

    public static int init_pool(ArrayList<String> allAddress){
        allAddress.forEach((item)->{
            String[] tmp = item.split(":");
            try{
                ManagedChannel channel = createChannel(tmp[0], Integer.parseInt(tmp[1]));
                pool.put(item, channel);
            }
            catch (Exception ex){
                LOGGER.error(ex);
            }
        });
        return ReturnCode.OK;
    }

    private static ManagedChannel createChannel(String ip, int port) throws Exception{
        for(int i=0; i<3; i++){
            try{
                LOGGER.info("try to connect {}:{}", ip, port);
                ManagedChannel channel =  ManagedChannelBuilder.forAddress(ip, port).usePlaintext().build();
                LOGGER.info(channel);
                return channel;
            }
            catch (Exception ex){
                LOGGER.info("try create channel: {}:{}", ip, port);
                TimeUnit.SECONDS.sleep(5);
                return null;
            }
        }
        return null;
    }

    public static ManagedChannel getChannel(String address){
        return pool.get(address);
    }
}
