package org.fedai.osx.broker.ptp;

import com.google.inject.Singleton;
import org.fedai.osx.broker.pojo.RouterTableSetRequest;
import org.fedai.osx.broker.pojo.RouterTableSetResponse;
import org.fedai.osx.broker.pojo.SetSelfPartyRequest;
import org.fedai.osx.broker.pojo.SetSelfPartyResponse;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;

import javax.inject.Inject;

@Singleton
@Register(uris ={UriConstants.HTTP_SET_ROUTER  },allowInterUse = false)
public class SelfPartySetService extends AbstractServiceAdaptorNew<SetSelfPartyRequest, SetSelfPartyResponse> {

    @Inject
    RouterServiceRegister  routerServiceRegister;

    @Override
    protected SetSelfPartyResponse doService(OsxContext context, SetSelfPartyRequest data) {

        RouterService routerService =routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);

        routerService.setSelfPartyIds(data.getSelfPartys());
        return null;
    }

    @Override
    protected SetSelfPartyResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return null;
    }

    @Override
    public SetSelfPartyRequest decode(Object object) {
        return null;
    }
}






