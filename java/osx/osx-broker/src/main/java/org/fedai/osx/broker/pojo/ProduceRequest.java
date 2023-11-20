package org.fedai.osx.broker.pojo;

import lombok.Data;


@Data
public class ProduceRequest {
    String topic;
    byte[] payload;
    String msgCode = "";
    String msgFlag = "";

}
