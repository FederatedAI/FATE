package org.fedai.osx.broker.security;



import org.fedai.osx.core.context.OsxContext;

public interface TokenValidator {
    public void validate(OsxContext context, String token);
}
