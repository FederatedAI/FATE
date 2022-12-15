package com.osx.core.flow;




import com.osx.core.token.TokenResult;

import java.util.Collection;

public interface TokenService {

    TokenResult requestToken(String resource, int acquireCount, boolean prioritized);


    /**
     * Request tokens from remote token server.
     *
     * @param ruleId the unique rule ID
     * @param acquireCount token count to acquire
     * @param prioritized whether the request is prioritized
     * @return result of the token request
     */
    //TokenResult requestToken(Long ruleId, int acquireCount, boolean prioritized);

    /**
     * Request tokens for a specific parameter from remote token server.
     *
     * @param ruleId the unique rule ID
     * @param acquireCount token count to acquire
     * @param params parameter list
     * @return result of the token request
     */
    //TokenResult requestParamToken(Long ruleId, int acquireCount, Collection<Object> params);

    /**
     * Request acquire concurrent tokens from remote token server.
     *
     * @param clientAddress the address of the request belong.
     * @param ruleId ruleId the unique rule ID
     * @param acquireCount token count to acquire
     * @return result of the token request
     */
    //TokenResult requestConcurrentToken(String clientAddress,Long ruleId,int acquireCount);
    /**
     * Request release concurrent tokens from remote token server asynchronously.
     *
     * @param tokenId the unique token ID
     */
    void releaseConcurrentToken(Long tokenId);
}
