package org.fedai.osx.broker.pojo;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.router.RouterInfo;

@Data
public class RouterSetRequest {

    public static class  BooleanFilter{
        @Override
        public boolean equals(Object obj) {
            if(obj instanceof Boolean){
                boolean result=   !(Boolean)obj;
                return result;
            }
            return  true;
        }
    }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Protocol protocol;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String desPartyId;
    @JsonIgnore
    private String desRole;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String url;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String ip;
    private Integer port;
    @JsonInclude(value = JsonInclude.Include.CUSTOM,valueFilter = RouterInfo.BooleanFilter.class)
    private boolean useSSL = false;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String certChainFile;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String privateKeyFile;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String caFile;
    @JsonInclude(value = JsonInclude.Include.CUSTOM,valueFilter = RouterInfo.BooleanFilter.class)
    private boolean useKeyStore = false ;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String keyStoreFilePath;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String keyStorePassword;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String trustStoreFilePath;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private String trustStorePassword;

}
