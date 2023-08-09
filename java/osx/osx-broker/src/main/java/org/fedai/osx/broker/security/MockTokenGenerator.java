package org.fedai.osx.broker.security;


import org.fedai.osx.api.context.Context;

public class MockTokenGenerator implements TokenGenerator{


    @Override
    public String createNewToken(Context context) {
        return "mock";
    }
}
