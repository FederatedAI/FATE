package org.fedai.osx.broker.security;


import org.fedai.osx.api.context.Context;

public interface TokenValidator {
    public void validate(Context context, String token);
}
