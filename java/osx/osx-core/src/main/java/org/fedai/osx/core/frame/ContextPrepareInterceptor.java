package org.fedai.osx.core.frame;

import io.grpc.*;
import io.grpc.stub.MetadataUtils;
import jdk.nashorn.internal.runtime.URIUtils;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.utils.UrlUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;

public class ContextPrepareInterceptor implements ServerInterceptor ,ClientInterceptor{

    Logger logger = LoggerFactory.getLogger(ContextPrepareInterceptor.class);

    static final ContextPrepareInterceptor INTERCEPTOR = new ContextPrepareInterceptor();
    public static  Context.Key<Object> CONTEXTKEY_SOURCEIP = Context.key("sourceIp");
    public static  Context.Key<String> CONTEXTKEY_VERSION = Context.key(PtpHttpHeader.Version);
    public static  Context.Key<String> CONTEXTKEY_TECH_PROVIDER  =  Context.key(PtpHttpHeader.TechProviderCode);
    public static  Context.Key<String> CONTEXTKEY_TRACE_ID = Context.key(PtpHttpHeader.TraceID);
    public static  Context.Key<String>  CONTEXTKEY_TOKEN = Context.key(PtpHttpHeader.Token);
    public static  Context.Key<String>  CONTEXTKEY_URI =Context.key(PtpHttpHeader.Uri);
    public static  Context.Key<String>  CONTEXTKEY_FROM_NODE_ID =Context.key(PtpHttpHeader.FromNodeID);
    public static  Context.Key<String>  CONTEXTKEY_FROM_INST_ID =Context.key(PtpHttpHeader.FromInstID);
    public static  Context.Key<String>  CONTEXTKEY_TARGET_NODE_ID =Context.key(PtpHttpHeader.TargetNodeID);
    public static  Context.Key<String>  CONTEXTKEY_TARGET_INST_ID =Context.key(PtpHttpHeader.TargetInstID);
    public static  Context.Key<String>  CONTEXTKEY_SESSION_ID =Context.key(PtpHttpHeader.SessionID);
    public static  Context.Key<String>  CONTEXTKEY_TOPIC_KEY =Context.key(PtpHttpHeader.MessageTopic);
    public static  Context.Key<String>  CONTEXTKEY_QUEUE_TYPE =Context.key(PtpHttpHeader.QueueType);
    public static  Context.Key<String>  CONTEXTKEY_MSG_FLAG =Context.key(PtpHttpHeader.MessageFlag);

//    public static  Context.Key<String>  CONTEXTKEY_TIMEOUT =Context.key("x-ptp-timeout");

    public static final Metadata.Key<String> METAKEY_TRACE_ID = Metadata.Key.of(CONTEXTKEY_TRACE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_FROM_INST_ID = Metadata.Key.of(CONTEXTKEY_FROM_INST_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_FROM_NODE_ID = Metadata.Key.of(CONTEXTKEY_FROM_NODE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_VERSION = Metadata.Key.of(CONTEXTKEY_VERSION.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_TECH_PROVIDER_CODE = Metadata.Key.of(CONTEXTKEY_TECH_PROVIDER.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_TOKEN = Metadata.Key.of(CONTEXTKEY_TOKEN.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_TARGET_NODE_ID = Metadata.Key.of(CONTEXTKEY_TARGET_NODE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_TARGET_INST_ID = Metadata.Key.of(CONTEXTKEY_TARGET_INST_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_SESSION_ID = Metadata.Key.of(CONTEXTKEY_SESSION_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_TOPIC_KEY = Metadata.Key.of(CONTEXTKEY_TOPIC_KEY.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String> METAKEY_URI = Metadata.Key.of(CONTEXTKEY_URI.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String>  METAKEY_QUEUE_TYPE =Metadata.Key.of(CONTEXTKEY_QUEUE_TYPE.toString(), Metadata.ASCII_STRING_MARSHALLER);
    public static final Metadata.Key<String>  METAKEY_MSG_FLAG =Metadata.Key.of(CONTEXTKEY_MSG_FLAG.toString(), Metadata.ASCII_STRING_MARSHALLER);


    private void setMetadata(Metadata metadata, Metadata.Key<String> key, String v) {

        if (StringUtils.isNotEmpty(v)) {
            logger.info("======meta data put {} {}",key,v);
            metadata.put(key, v);
        }
    }


    public static <T> boolean required(T... inputs) {
        System.err.println("ooooooooo"+inputs);
        if (null == inputs || inputs.length < 1) {
            return false;
        }
        for (T input : inputs) {
            if (null == input) {
                return false;
            }
            if (input instanceof CharSequence) {
                return !((CharSequence) input).chars().allMatch(Character::isWhitespace);
            }
            if (input instanceof Collection) {
                return !((Collection<?>) input).isEmpty();
            }
            if (input instanceof Map) {
                return !((Map<?, ?>) input).isEmpty();
            }
            if (input.getClass().isArray()) {
                return Array.getLength(input) > 0;
            }
        }
        return true;
    }

//    MESH_VERSION("x-ptp-version"),
//    MESH_TECH_PROVIDER_CODE("x-ptp-tech-provider-code"),
//    MESH_TRACE_ID("x-ptp-trace-id"),
//    MESH_TOKEN("x-ptp-token"),
//    MESH_URI("x-ptp-uri"),
//    MESH_FROM_NODE_ID("x-ptp-from-node-id"),
//    MESH_FROM_INST_ID("x-ptp-from-inst-id"),
//    MESH_TARGET_NODE_ID("x-ptp-target-node-id"),
//    MESH_TARGET_INST_ID("x-ptp-target-inst-id"),
//    MESH_SESSION_ID("x-ptp-session-id"),
//    MESH_TOPIC("x-ptp-topic"),
//    MESH_TIMEOUT("x-ptp-timeout"),

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata metadata, ServerCallHandler<ReqT, RespT> next){

        String remoteAddr = call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
        String[] remoteAddrSplited = remoteAddr.split(":");
        String remoteIp = remoteAddrSplited[0].replaceAll("\\/", "");
        logger.info("receive metadata {} {} {}",metadata,Optional.ofNullable(metadata.get(METAKEY_TRACE_ID)).orElse(""),remoteIp);
        Context context = Context.current()
                .withValue(CONTEXTKEY_TRACE_ID, Optional.ofNullable(metadata.get(METAKEY_TRACE_ID)).orElse(""))
                .withValue(CONTEXTKEY_FROM_INST_ID, metadata.get(METAKEY_FROM_INST_ID))
                .withValue(CONTEXTKEY_FROM_NODE_ID, metadata.get(METAKEY_FROM_NODE_ID))
                .withValue(CONTEXTKEY_VERSION, metadata.get(METAKEY_VERSION))
//                .withValue(tim metadata.get(GrpcContextKey.TIMESTAMP))
                .withValue(CONTEXTKEY_SOURCEIP, remoteIp)
                .withValue(CONTEXTKEY_TECH_PROVIDER, metadata.get(METAKEY_TECH_PROVIDER_CODE))
                .withValue(CONTEXTKEY_TOKEN, metadata.get(METAKEY_TOKEN))
                .withValue(CONTEXTKEY_TARGET_NODE_ID, metadata.get(METAKEY_TARGET_NODE_ID))
                .withValue(CONTEXTKEY_TARGET_INST_ID, metadata.get(METAKEY_TARGET_INST_ID))
                .withValue(CONTEXTKEY_TOPIC_KEY,metadata.get(METAKEY_TOPIC_KEY))
                .withValue(CONTEXTKEY_SESSION_ID, metadata.get(METAKEY_SESSION_ID))
                .withValue(CONTEXTKEY_URI,metadata.get(METAKEY_URI))
                .withValue(CONTEXTKEY_QUEUE_TYPE,metadata.get(METAKEY_QUEUE_TYPE))
                .withValue(CONTEXTKEY_MSG_FLAG,metadata.get(METAKEY_MSG_FLAG));






        return Contexts.interceptCall(context, call, metadata, next);
    }

    @Override
    public <I, O> ClientCall<I, O> interceptCall(MethodDescriptor<I, O> descriptor, CallOptions options, Channel channel) {
        Metadata metadata = new Metadata();
//        OsxContext  fateContext =OsxContext.getContextFromThreadLocal();
        OsxContext  osxContext =  OsxContext.getContextFromThreadLocal();
        setMetadata(metadata, METAKEY_TRACE_ID, Optional.ofNullable(osxContext.getTraceId()).orElse(""));
        setMetadata(metadata, METAKEY_FROM_INST_ID, Optional.ofNullable(osxContext.getSrcInstId()).orElse(""));
        setMetadata(metadata, METAKEY_FROM_NODE_ID, Optional.ofNullable(osxContext.getSrcNodeId()).orElse(""));
        setMetadata(metadata, METAKEY_VERSION, Optional.ofNullable(osxContext.getVersion()).orElse(""));
        setMetadata(metadata, METAKEY_TECH_PROVIDER_CODE, Optional.ofNullable(osxContext.getTechProviderCode()).orElse(""));
        setMetadata(metadata, METAKEY_TOKEN, Optional.ofNullable(osxContext.getToken()).orElse(""));
        setMetadata(metadata, METAKEY_TARGET_NODE_ID,Optional.ofNullable(osxContext.getDesNodeId()).orElse(""));
        setMetadata(metadata, METAKEY_TARGET_INST_ID, Optional.ofNullable(osxContext.getDesInstId()).orElse(""));
        setMetadata(metadata, METAKEY_SESSION_ID, Optional.ofNullable(osxContext.getSessionId()).orElse(""));
        setMetadata(metadata, METAKEY_TOPIC_KEY, Optional.ofNullable(osxContext.getTopic()).orElse(""));
        setMetadata(metadata,METAKEY_URI, Optional.ofNullable(UrlUtil.buildUrl("grpcs://","fedai.org",osxContext.getUri())).orElse(""));
        setMetadata(metadata,METAKEY_QUEUE_TYPE,Optional.ofNullable(osxContext.getQueueType()).orElse(""));
        setMetadata(metadata,METAKEY_MSG_FLAG,Optional.ofNullable(osxContext.getMessageFlag()).orElse(""));

      //  logger.info("========client intercept======{}",metadata);
        return MetadataUtils.newAttachHeadersInterceptor(metadata).interceptCall(descriptor, options, channel);
    }



//    @Override
//    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
//        String remoteAddr = call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
//        String[] remoteAddrSplited = remoteAddr.split(":");
//        String remoteIp = remoteAddrSplited[0].replaceAll("\\/", "");
//        Context context = Context.current().withValue(sourceIp, remoteIp)
//                .withValue()
//
//        return Contexts.interceptCall(context, call, headers, next);
//    }

}