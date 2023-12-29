package org.fedai.osx.core.config;

import lombok.Data;

@Data
public class TransferMeta {

    String srcPartyId;
    String desPartyId;
    String srcRole;
    String desRole;
    String sessionId;
    String topic;


}
