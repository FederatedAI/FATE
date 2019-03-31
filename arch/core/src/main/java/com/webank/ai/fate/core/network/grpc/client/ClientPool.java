/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.core.network.grpc.client;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import com.webank.ai.fate.core.constant.StatusCode;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ClientPool {
    //TODO: channel pool
    private static final Logger LOGGER = LogManager.getLogger();
    private static ConcurrentHashMap<String, ManagedChannel> pool = new ConcurrentHashMap();

    public static int init_pool(ArrayList<String> servers){
        servers.forEach((server)->{
            if (createChannel(server) > 0){
                LOGGER.error("Failed to create channel for {}", server);
            }
        });
        return StatusCode.OK;
    }

    private static int createChannel(String address){
        String[] ipPort = address.split(":");
        String ip = ipPort[0];
        int port = Integer.parseInt(ipPort[1]);
        for(int i=0; i<3; i++){
            try{
                ManagedChannel channel =  ManagedChannelBuilder.forAddress(ip, port).usePlaintext().build();
                pool.put(address, channel);
                return StatusCode.OK;
            }
            catch (Exception ex){
                LOGGER.warn("Try to create channel again for {}:{}", ip, port);
                try{
                    TimeUnit.SECONDS.sleep(5);
                }
                catch (InterruptedException ex2){
                    LOGGER.warn(ex2);
                }
            }
        }
        return StatusCode.NETWORKERROR;
    }

    public static ManagedChannel getChannel(String address){
        return pool.get(address);
    }

    public static int rebuildChannel(String address){
        return createChannel(address);
    }
}
