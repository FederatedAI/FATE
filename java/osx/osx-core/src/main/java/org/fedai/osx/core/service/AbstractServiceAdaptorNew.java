package org.fedai.osx.core.service;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.grpc.stub.AbstractStub;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;

/**
 * @Description 默认的服务适配器
 * @Author
 **/

public abstract class AbstractServiceAdaptorNew<req, resp> implements ServiceAdaptorNew<req, resp> {

    protected String serviceName;
    Logger logger = LoggerFactory.getLogger(this.getClass().getName());
    //    ServiceAdaptor< req, resp> serviceAdaptor;
    InterceptorChainNew<req, resp> preChain = new DefaultInterceptorChainNew<>();
    InterceptorChainNew<req, resp> postChain = new DefaultInterceptorChainNew<>();
    private Map<String, Method> methodMap = Maps.newHashMap();
    private AbstractStub serviceStub;

    public AbstractServiceAdaptorNew() {

    }

    public void registerMethod(String actionType, Method method) {
        this.methodMap.put(actionType, method);
    }


    public Map<String, Method> getMethodMap() {
        return methodMap;
    }

    public void setMethodMap(Map<String, Method> methodMap) {
        this.methodMap = methodMap;
    }

    public AbstractServiceAdaptorNew<req, resp> addPreProcessor(InterceptorNew interceptor) {
        preChain.addInterceptor(interceptor);
        return this;
    }

    public void addPostProcessor(InterceptorNew interceptor) {
        postChain.addInterceptor(interceptor);
    }


    public AbstractStub getServiceStub() {
        return serviceStub;
    }

    public void setServiceStub(AbstractStub serviceStub) {
        this.serviceStub = serviceStub;
    }

    public String getServiceName() {
        return serviceName;
    }

    public void setServiceName(String serviceName) {
        this.serviceName = serviceName;
    }

    protected abstract resp doService(OsxContext context, req data);

    /**
     * @param context
     * @param data
     * @return
     * @throws Exception
     */
    @Override
    public resp service(OsxContext context, req data) throws RuntimeException {
        resp result = null;

        // context.preProcess();
        List<Throwable> exceptions = Lists.newArrayList();
        context.setReturnCode(StatusCode.PTP_SUCCESS);
//        if (!isOpen) {
//            return this.serviceFailInner(context, data, new ShowDownRejectException());
//        }
//        if (data.getBody() != null) {
//            context.putData(Dict.INPUT_DATA, data.getBody());
//        }

        try {


            context.setServiceName(this.serviceName);
            try {
                preChain.doProcess(context, data, null);
                result = doService(context, data);
            } catch (Throwable e) {
                exceptions.add(e);
                e.printStackTrace();
                logger.error("do service fail, {} ", ExceptionUtils.getStackTrace(e));
            }

        } catch (Throwable e) {
            exceptions.add(e);
            logger.error("service error", e);
        } finally {

            try {
                if (exceptions.size() != 0) {
                    result = this.serviceFail(context, data, exceptions);
                }
            } finally {
//                if(context instanceof OsxContext )
//                {
//                    OsxContext fateContext =(OsxContext )context;
//                    if(fateContext.needPrintFlowLog()){
//                        FlowLogUtil.printFlowLog(context);
//                    }
//                }else {
//
//                    FlowLogUtil.printFlowLog(context);
//                }
            }
            //  int returnCode = context.getReturnCode();

//            if(outboundPackage.getData()!=null) {
//                context.putData(Dict.OUTPUT_DATA, outboundPackage.getData());
//            }
            // context.postProcess(data, outboundPackage);

        }
        try {
            postChain.doProcess(context, data, result);
        } catch (Exception e) {
            logger.error("service PostDoProcess error", e);
        }
        return result;
    }

//    protected void printFlowLog(ctx context) {
////        context.printFlowLog();
//        FlowLogUtil.printFlowLog(context);
//    }

    protected resp serviceFailInner(OsxContext context, req data, Throwable e) {

        ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
        context.setReturnCode(exceptionInfo.getCode());
        context.setReturnMsg(exceptionInfo.getMessage());
        resp rsp = transformExceptionInfo(context, exceptionInfo);
        return rsp;
    }

    @Override
    public resp serviceFail(OsxContext context, req data, List<Throwable> errors) throws RuntimeException {
        Throwable e = errors.get(0);
        logger.error("service fail ", e);
        return serviceFailInner(context, data, e);
    }

    protected abstract resp transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo);


}