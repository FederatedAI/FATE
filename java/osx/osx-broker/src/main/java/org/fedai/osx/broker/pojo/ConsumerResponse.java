package org.fedai.osx.broker.pojo;

import com.google.protobuf.ByteString;
import lombok.Data;
import org.fedai.osx.core.router.RouterInfo;
import org.ppc.ptp.Osx;

@Data
public class ConsumerResponse {
    String  code;
    String  msg="";
    byte[]  payload;
    boolean  needRedirect= false;
    RouterInfo redirectRouterInfo;
//    Osx.TransportOutbound   redirectResult;

    public Osx.TransportOutbound toTransportOutbound() {
            Osx.TransportOutbound.Builder builder = Osx.TransportOutbound.newBuilder();
            return builder.setCode(code).setMessage(msg).setPayload(ByteString.copyFrom(payload)).build();

    }
}
