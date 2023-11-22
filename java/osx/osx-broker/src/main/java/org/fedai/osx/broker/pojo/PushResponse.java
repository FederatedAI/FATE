package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class PushResponse {
    public PushResponse(String code, String msg){
        this.code = code;
        this.msg = msg;
    }
    String  code;
    String  msg;
}
