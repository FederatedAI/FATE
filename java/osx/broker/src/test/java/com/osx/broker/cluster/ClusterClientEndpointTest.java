//package com.firework.transfer.cluster;
//
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.core.bean.MetaInfo;
//import com.firework.remote.pojo.ApplyTransferQueueResponse;
//
//import com.firework.remote.pojo.HeartbeatResponse;
//import io.grpc.ManagedChannel;
//import org.junit.Before;
//import org.junit.Test;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//
//public class ClusterClientEndpointTest {
//
//    Logger logger = LoggerFactory.getLogger(ClusterClientEndpointTest.class);
//
//    ClusterClientEndpoint  clusterClientEndpoint;
//
//    @Before
//    public  void init (){
//        MetaInfo.PROPERTY_CLUSTER_MANAGER_ADDRESS="localhost:8888";
//        clusterClientEndpoint = new ClusterClientEndpoint();
//        clusterClientEndpoint.start();
//    }
//
//    @Test
//    public  void testApplyTransferQueue(){
//        ApplyTransferQueueResponse applyTransferQueueResponse =
//                clusterClientEndpoint.applyTransferQueue("testInstanceId","testTransferId","testSessionId");
//        logger.info("testApplyTransferQueue response  {}",applyTransferQueueResponse);
//
//    }
//
//    @Test
//    public  void  testHeartbeat(){
//        HeartbeatRequest  heartbeatRequest =  new  HeartbeatRequest();
//        heartbeatRequest.setInstanceId("testInstance");
//        heartbeatRequest.setIp("33333333");
//        HeartbeatResponse heartbeatResponse = clusterClientEndpoint.heartbeat(heartbeatRequest);
//    }
//
//
//}
