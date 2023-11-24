package org.fedai.osx.broker.pojo;

import lombok.Data;
import org.fedai.osx.core.utils.JsonUtil;

import java.nio.charset.StandardCharsets;


@Data
public class ProduceRequest implements SerializeAware{
    String topic;
    byte[] payload;
    String msgCode = "";
    String msgFlag = "";

    public  byte[] serialize(){
        return JsonUtil.object2Json(this).getBytes(StandardCharsets.UTF_8);
    }

}
