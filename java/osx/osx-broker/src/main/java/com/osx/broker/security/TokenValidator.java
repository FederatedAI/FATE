package com.osx.broker.security;


import com.osx.api.context.Context;

public interface TokenValidator {
    public void validate(Context context, String token);
}
