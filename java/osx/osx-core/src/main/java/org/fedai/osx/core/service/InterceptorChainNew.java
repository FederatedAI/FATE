package org.fedai.osx.core.service;


public interface InterceptorChainNew<req, resp> extends InterceptorNew<req, resp> {
    public void addInterceptor(InterceptorNew<req, resp> interceptor);
}
