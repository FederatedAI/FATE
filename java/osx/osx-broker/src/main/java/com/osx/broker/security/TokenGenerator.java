package com.osx.broker.security;


import com.osx.api.context.Context;

public interface TokenGenerator {

    String  createNewToken(Context context);

}
