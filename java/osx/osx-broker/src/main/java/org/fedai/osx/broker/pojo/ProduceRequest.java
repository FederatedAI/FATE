package org.fedai.osx.broker.pojo;

import lombok.Data;
import org.fedai.osx.core.constant.QueueType;


@Data
public class ProduceRequest {
    String  topic;
    byte[]  payload;
    String  msgCode="";
    String  msgFlag="";

}
