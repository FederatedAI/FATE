package com.osx.broker.security;


import com.osx.api.context.Context;

public class MockTokenGenerator implements TokenGenerator{


    @Override
    public String createNewToken(Context context) {
        return "mock";
    }
}
