package org.fedai.osx.broker.pojo;

import lombok.Data;

@Data
public class HttpInvokeResult {
    byte[] payload;                  // 报文，上层通信内容承载，序列化协议由上层基于SPI可插拔
    String code ;                 // 状态码
    String  message ;
}
