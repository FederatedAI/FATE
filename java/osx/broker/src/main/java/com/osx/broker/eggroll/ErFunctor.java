package com.osx.broker.eggroll;

import com.google.protobuf.ByteString;
import com.webank.eggroll.core.meta.Meta;

import java.util.Map;

public class ErFunctor extends BaseProto<Meta.Functor> {

    String name;
    String serdes;
    byte[] body;
    Map<String, String> options;
    public ErFunctor(String name, String serdes, byte[] body, Map<String, String> options) {
        this.name = name;
        this.serdes = serdes;
        this.body = body;
        this.options = options;

    }

    public static ErFunctor parseFromPb(Meta.Functor functor) {
        if (functor == null)
            return null;
        String name = functor.getName();
        ByteString bodyByteString = functor.getBody();
        Map<String, String> options = functor.getOptionsMap();
        String serdes = functor.getSerdes();
        ErFunctor erFunctor = new ErFunctor(name, serdes, bodyByteString != null ? bodyByteString.toByteArray() : null, options);
        return erFunctor;
    }

    @Override
    Meta.Functor toProto() {
        return Meta.Functor.newBuilder().
                setName(this.name).
                setSerdes(this.serdes).putAllOptions(options).
                setBody(ByteString.copyFrom(body)).build();
    }
}

//case class ErFunctor(name: String = StringConstants.EMPTY,
//                     serdes: String = StringConstants.EMPTY,
//                     body: Array[Byte]) extends RpcMessage
