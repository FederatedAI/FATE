package org.fedai.osx.broker.token;

import java.util.Properties;

public interface TokenValidator {
    public void  init(String name ,Properties properties);
    public void  validate(String token);
}
