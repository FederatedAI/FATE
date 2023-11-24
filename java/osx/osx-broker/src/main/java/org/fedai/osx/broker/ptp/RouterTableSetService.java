package org.fedai.osx.broker.ptp;

import com.google.common.base.Preconditions;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.pojo.RouterAddRequest;
import org.fedai.osx.broker.pojo.RouterAddResponse;
import org.fedai.osx.broker.pojo.RouterTableSetRequest;
import org.fedai.osx.broker.pojo.RouterTableSetResponse;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.ppc.ptp.Osx;

@Singleton
@Register(uris ={UriConstants.HTTP_SET_ROUTER  ,},allowInterUse = false)
public class RouterTableSetService extends AbstractServiceAdaptorNew<RouterTableSetRequest, RouterTableSetResponse> {

    @Inject
    RouterServiceRegister  routerServiceRegister;

    @Override
    protected RouterTableSetResponse doService(OsxContext context, RouterTableSetRequest data) {
        RouterService routerService = routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        Preconditions.checkArgument(data!=null&& StringUtils.isNotEmpty(data.getData()));
        routerService.setRouterTable(data.getData());
        return new  RouterTableSetResponse();
    }

    @Override
    protected RouterTableSetResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return null;
    }

    @Override
    public RouterTableSetRequest decode(Object object) {

        
        return null;
    }


}
