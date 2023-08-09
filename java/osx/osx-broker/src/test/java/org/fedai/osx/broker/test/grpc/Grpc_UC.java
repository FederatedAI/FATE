package org.fedai.osx.broker.test.grpc;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import org.fedai.osx.api.router.RouterInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.exceptions.RemoteRpcException;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.utils.JsonUtil;
import org.junit.Test;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

public class Grpc_UC {

    String contextStr = "{\"actionType\":\"unary-call-new\",\"protocol\":\"grpc\",\"techProviderCode\":\"FATE\",\"needCheckRouterInfo\":true,\"costTime\":0,\"resourceName\":\"I_unary-call-new\",\"timeStamp\":1685499290484,\"downstreamCost\":0,\"downstreamBegin\":0,\"destination\":false,\"sourceIp\":\"127.0.0.1\",\"desPartyId\":\"20008\",\"srcPartyId\":\"\",\"returnCode\":\"0\",\"desComponent\":\"fateflow\",\"routerInfo\":{\"protocol\":\"grpc\",\"sourcePartyId\":\"\",\"desPartyId\":\"20008\",\"desRole\":\"fateflow\",\"url\":\"\",\"host\":\"127.0.0.1\",\"port\":9360,\"useSSL\":false,\"negotiationType\":\"\",\"certChainFile\":\"\",\"privateKeyFile\":\"\",\"caFile\":\"\",\"resource\":\"-20008\",\"cycle\":false},\"selfPartyId\":\"10008\"}";
    String routerJson = "{\n" +
            "    \"protocol\": \"grpc\",\n" +
            "    \"sourcePartyId\": \"\",\n" +
            "    \"desPartyId\": \"10008\",\n" +
            "    \"desRole\": \"fateflow\",\n" +
            "    \"url\": \"http://127.0.0.1:8087/osx/inbound\",\n" +
            "    \"host\": \"127.0.0.1\",\n" +
            "    \"port\": 9883,\n" +
            "    \"useSSL\": true,\n" +
            "    \"negotiationType\": \"TLS\",\n" +
            "    \"certChainFile\": \"D:/33/127.0.0.1.crt\",\n" +
            "    \"privateKeyFile\": \"D:/33/127.0.0.1.key\",\n" +
            "    \"caFile\": \"D:/33/testRoot.crt\",\n" +
            "    \"resource\": \"-10008\",\n" +
            "    \"cycle\": false\n" +
            "}";

    @Test
    public void run(){
        FateContext context = JsonUtil.json2Object(contextStr,FateContext.class);
        RouterInfo routerInfo = JsonUtil.json2Object(routerJson,RouterInfo.class);
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = null;
        if (context.getData(Dict.BLOCKING_STUB) == null) {
            ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo, true);
            stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        } else {
            stub = (PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub) context.getData(Dict.BLOCKING_STUB);
        }
        try {
            //  logger.info("===========send data {}",produceRequest);
            Osx.Outbound invoke = stub.invoke(null);
        } catch (StatusRuntimeException e) {
            e.printStackTrace();
            throw new RemoteRpcException(StatusCode.NET_ERROR, "send to " + routerInfo.toKey() + " error : " + e.getMessage());
        }
    }
}
