package org.fedai.osx.broker.http;

import lombok.Data;

import java.util.Map;

@Data
public class HttpDataWrapper {
    Map<String, String> headers;
    String mime;
    byte[] payload;
}
