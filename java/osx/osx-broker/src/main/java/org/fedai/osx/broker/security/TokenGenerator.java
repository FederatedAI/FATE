package org.fedai.osx.broker.security;


import org.fedai.osx.api.context.Context;

public interface TokenGenerator {

    String  createNewToken(Context context);

}
