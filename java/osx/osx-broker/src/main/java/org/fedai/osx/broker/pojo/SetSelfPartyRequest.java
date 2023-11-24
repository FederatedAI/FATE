package org.fedai.osx.broker.pojo;

import lombok.Data;

import java.util.Set;

@Data
public class SetSelfPartyRequest {

    Set<String> selfPartys;

}
