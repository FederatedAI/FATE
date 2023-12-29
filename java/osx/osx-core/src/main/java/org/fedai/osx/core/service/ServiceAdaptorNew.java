package org.fedai.osx.core.service;


import org.fedai.osx.core.context.OsxContext;
import org.ppc.ptp.Osx;

import java.util.List;

public interface ServiceAdaptorNew<req, rsp> {

    public rsp service(OsxContext context, req inboundPackage);

    public rsp serviceFail(OsxContext context, req data, List<Throwable> e);

    default public req decode(Object object){return null;};

    default  public Osx.Outbound toOutbound(rsp response){return null;};

}
