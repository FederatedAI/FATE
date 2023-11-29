package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class ProduceResponse {

    String code;
    String msg;
    public ProduceResponse(String code, String msg) {
        this.code = code;
        this.msg = msg;
    }
}
