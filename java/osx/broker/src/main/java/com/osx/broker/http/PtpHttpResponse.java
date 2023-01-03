package com.osx.broker.http;

import lombok.Data;

import java.util.Map;
@Data
public class PtpHttpResponse {
    Map header;
    String  code;
    String  message;
    byte[]  payload;
}
