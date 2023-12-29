package org.fedai.osx.broker.security;


import org.fedai.osx.core.context.OsxContext;

public class MockTokenGenerator implements TokenGenerator {


    @Override
    public String createNewToken(OsxContext context) {
        return "mock";
    }
}
