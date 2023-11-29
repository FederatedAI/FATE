package org.fedai.osx.broker.security;


import org.fedai.osx.core.context.OsxContext;

public interface TokenGenerator {

    String createNewToken(OsxContext context);

}
