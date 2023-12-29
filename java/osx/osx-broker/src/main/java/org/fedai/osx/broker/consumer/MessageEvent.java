package org.fedai.osx.broker.consumer;

import lombok.Data;

@Data
public class MessageEvent {
    String srcPartyId;
    String desPartyId;
    String srcComponent;
    String desComponent;
    String topic;
    String sessionId;
}
