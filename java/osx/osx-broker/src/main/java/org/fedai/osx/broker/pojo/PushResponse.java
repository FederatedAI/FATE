package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class PushResponse {
    String code;
    String msg;
    public PushResponse(String code, String msg) {
        this.code = code;
        this.msg = msg;
    }
}
