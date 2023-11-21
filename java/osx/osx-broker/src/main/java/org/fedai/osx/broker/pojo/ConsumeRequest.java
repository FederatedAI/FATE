package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class ConsumeRequest {

    public ConsumeRequest(){

    }
    public ConsumeRequest(String topic,int timeout,boolean needBlock){
        this.needBlock = needBlock;
        this.topic = topic;
        this.timeout = timeout;
    }

    boolean  needBlock;
    String  topic;
    int  timeout;
}
