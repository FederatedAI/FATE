package com.osx.broker.grpc;
import com.osx.core.context.Context;
import com.osx.broker.constants.Direction;
import com.osx.broker.util.ResourceUtil;
//import com.firework.transfer.service.TokenApplyService;
import com.google.common.base.Preconditions;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ForwardPullRespSO implements StreamObserver<Proxy.Packet>{

    Logger logger = LoggerFactory.getLogger(ForwardPullRespSO.class);

    //TokenApplyService tokenApplyService;

    Context context;

    StreamObserver<Proxy.Packet>  backStreamObserver;

    public ForwardPullRespSO(Context  context ,StreamObserver<Proxy.Packet>  backStreamObserver){
            //,TokenApplyService  tokenApplyService){

       // Preconditions.checkArgument(tokenApplyService!=null);
        Preconditions.checkArgument(backStreamObserver!=null);
        Preconditions.checkArgument(context!=null);
        this.context = context;
       // this.tokenApplyService = tokenApplyService;
        this.backStreamObserver = backStreamObserver;
    }

    @Override
    public void onNext(Proxy.Packet value) {
        String   resource = ResourceUtil.buildResource(context.getRouterInfo(), Direction.DOWN);
        //tokenApplyService.applyToken(context,resource,value.toByteArray().length);
        backStreamObserver.onNext(value);
    }

    @Override
    public void onError(Throwable t) {
        logger.error("error",t);
        t.printStackTrace();
        backStreamObserver.onError(t);
    }

    @Override
    public void onCompleted() {
        backStreamObserver.onCompleted();
    }
}
