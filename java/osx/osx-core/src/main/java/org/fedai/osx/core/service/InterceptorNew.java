package org.fedai.osx.core.service;


import org.fedai.osx.core.context.OsxContext;

public interface InterceptorNew<req, resp> {

    default public void doProcess(OsxContext context, req inbound, resp outbound) throws Exception {

    }

}
