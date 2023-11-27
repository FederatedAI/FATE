package org.fedai.osx.broker.token;

import org.fedai.osx.core.exceptions.InvalidRequestException;

import java.util.Properties;

public class SimpleTokenValidator implements TokenValidator{

    String rightToken = "";

    @Override
    public void init(String name ,Properties properties) {
        rightToken= properties.getProperty(name+".token");
    }

    @Override
    public void validate(String token) {
        if(token==null||!token.equals(rightToken)){
            throw new  InvalidRequestException("token is invalid");
        }
    }
}
