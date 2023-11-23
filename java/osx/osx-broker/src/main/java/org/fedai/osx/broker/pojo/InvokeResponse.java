package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class InvokeResponse {

//    map<string, string>  metadata = 1;  // 报头，可选，预留扩展，Dict，序列化协议由通信层统一实现
//    bytes payload = 2;                  // 报文，上层通信内容承载，序列化协议由上层基于SPI可插拔
//    string code = 3;                    // 状态码
//    string message = 4;

    String  code;
    String  message;
    byte[]  payload;

}
