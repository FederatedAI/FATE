package com.osx.broker.eggroll;

import lombok.Data;

@Data
public class MessageEvent {
    String topic;
    int  index;
}
