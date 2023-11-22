//package org.fedai.osx.broker.grpc;
//
//import io.grpc.*;
//import io.grpc.stub.MetadataUtils;
//
//import java.lang.reflect.Array;
//import java.util.Collection;
//import java.util.Map;
//import java.util.Optional;
//
//public class ContextPrepareInterceptor implements ServerInterceptor ,ClientInterceptor{
//
//    static final ContextPrepareInterceptor INTERCEPTOR = new ContextPrepareInterceptor();
//
//    public static  Context.Key<Object> CONTEXTKEY_SOURCEIP = Context.key("sourceIp");
//    public static  Context.Key<String> CONTEXTKEY_VERSION = Context.key("x-ptp-version");
//    public static  Context.Key<String> CONTEXTKEY_TECH_PROVIDER  =  Context.key("x-ptp-tech-provider-code");
//    public static  Context.Key<String> CONTEXTKEY_TRACE_ID = Context.key("x-ptp-trace-id");
//    public static  Context.Key<String>  CONTEXTKEY_TOKEN = Context.key("x-ptp-token");
//    public static  Context.Key<String>  CONTEXTKEY_URI =Context.key("x-ptp-uri");
//    public static  Context.Key<String>  CONTEXTKEY_FROM_NODE_ID =Context.key("x-ptp-from-node-id");
//    public static  Context.Key<String>  CONTEXTKEY_FROM_INST_ID =Context.key("x-ptp-from-inst-id");
//    public static  Context.Key<String>  CONTEXTKEY_TARGET_NODE_ID =Context.key("x-ptp-target-node-id");
//    public static  Context.Key<String>  CONTEXTKEY_TARGET_INST_ID =Context.key("x-ptp-target-inst-id");
//    public static  Context.Key<String>  CONTEXTKEY_SESSION_ID =Context.key("x-ptp-session-id");
//    public static  Context.Key<String>  CONTEXTKEY_TOPIC_KEY =Context.key("x-ptp-topic");
////    public static  Context.Key<String>  CONTEXTKEY_TIMEOUT =Context.key("x-ptp-timeout");
//
//
//
//    public static final Metadata.Key<String> TRACE_ID = Metadata.Key.of(CONTEXTKEY_TRACE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> FROM_INST_ID = Metadata.Key.of(CONTEXTKEY_FROM_INST_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> FROM_NODE_ID = Metadata.Key.of(CONTEXTKEY_FROM_NODE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> VERSION = Metadata.Key.of(CONTEXTKEY_VERSION.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> TECH_PROVIDER_CODE = Metadata.Key.of(CONTEXTKEY_TECH_PROVIDER.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> TOKEN = Metadata.Key.of(CONTEXTKEY_TOKEN.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> TARGET_NODE_ID = Metadata.Key.of(CONTEXTKEY_TARGET_NODE_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> TARGET_INST_ID = Metadata.Key.of(CONTEXTKEY_TARGET_INST_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//    public static final Metadata.Key<String> SESSION_ID = Metadata.Key.of(CONTEXTKEY_SESSION_ID.toString(), Metadata.ASCII_STRING_MARSHALLER);
//
//
//
//    private void setMetadata(Metadata metadata, Metadata.Key<String> key, String v) {
//        if (required(v)) {
//            metadata.put(key, v);
//        }
//    }
//
//
//    public static <T> boolean required(T... inputs) {
//        if (null == inputs || inputs.length < 1) {
//            return false;
//        }
//        for (T input : inputs) {
//            if (null == input) {
//                return false;
//            }
//            if (input instanceof CharSequence) {
//                return !((CharSequence) input).chars().allMatch(Character::isWhitespace);
//            }
//            if (input instanceof Collection) {
//                return !((Collection<?>) input).isEmpty();
//            }
//            if (input instanceof Map) {
//                return !((Map<?, ?>) input).isEmpty();
//            }
//            if (input.getClass().isArray()) {
//                return Array.getLength(input) > 0;
//            }
//        }
//        return true;
//    }
//
////    MESH_VERSION("x-ptp-version"),
////    MESH_TECH_PROVIDER_CODE("x-ptp-tech-provider-code"),
////    MESH_TRACE_ID("x-ptp-trace-id"),
////    MESH_TOKEN("x-ptp-token"),
////    MESH_URI("x-ptp-uri"),
////    MESH_FROM_NODE_ID("x-ptp-from-node-id"),
////    MESH_FROM_INST_ID("x-ptp-from-inst-id"),
////    MESH_TARGET_NODE_ID("x-ptp-target-node-id"),
////    MESH_TARGET_INST_ID("x-ptp-target-inst-id"),
////    MESH_SESSION_ID("x-ptp-session-id"),
////    MESH_TOPIC("x-ptp-topic"),
////    MESH_TIMEOUT("x-ptp-timeout"),
//
//    @Override
//    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata metadata, ServerCallHandler<ReqT, RespT> next){
//        Context context = Context.current()
//                .withValue(CONTEXTKEY_TRACE_ID, Optional.ofNullable(metadata.get(TRACE_ID)).orElse(""))
//                .withValue(CONTEXTKEY_FROM_INST_ID, metadata.get(FROM_INST_ID))
//                .withValue(CONTEXTKEY_FROM_NODE_ID, metadata.get(FROM_NODE_ID))
//                .withValue(CONTEXTKEY_VERSION, metadata.get(VERSION))
////                .withValue(tim metadata.get(GrpcContextKey.TIMESTAMP))
//
//                .withValue(CONTEXTKEY_TECH_PROVIDER, metadata.get(TECH_PROVIDER_CODE))
//                .withValue(CONTEXTKEY_TOKEN, metadata.get(TOKEN))
//                .withValue(CONTEXTKEY_TARGET_NODE_ID, metadata.get(TARGET_NODE_ID))
//                .withValue(CONTEXTKEY_TARGET_INST_ID, metadata.get(TARGET_INST_ID))
//                .withValue(CONTEXTKEY_SESSION_ID, metadata.get(SESSION_ID));
//        return Contexts.interceptCall(context, call, metadata, next);
//    }
//
//    @Override
//    public <I, O> ClientCall<I, O> interceptCall(MethodDescriptor<I, O> descriptor, CallOptions options, Channel channel) {
//        Metadata metadata = new Metadata();
////        setMetadata(metadata, CONTEXTKEY_TRACE_ID, Mesh.context().getTraceId());
////        setMetadata(metadata, CONTEXTKEY_FROM_INST_ID, Optional.ofNullable(Mesh.context().getConsumer().getInstId()).orElse(""));
////        setMetadata(metadata, CONTEXTKEY_FROM_NODE_ID, Optional.ofNullable(Mesh.context().getConsumer().getNodeId()).orElse(""));
////        setMetadata(metadata, CONTEXTKEY_VERSION, com.be.mesh.client.prsim.Context.Metadata.MESH_VERSION.get());
////        setMetadata(metadata, CONTEXTKEY_TECH_PROVIDER, com.be.mesh.client.prsim.Context.Metadata.MESH_TECH_PROVIDER_CODE.get());
////        setMetadata(metadata, CONTEXTKEY_TOKEN, com.be.mesh.client.prsim.Context.Metadata.MESH_TOKEN.get());
////        setMetadata(metadata, CONTEXTKEY_TARGET_NODE_ID, com.be.mesh.client.prsim.Context.Metadata.MESH_TARGET_NODE_ID.get());
////        setMetadata(metadata, CONTEXTKEY_TARGET_INST_ID, com.be.mesh.client.prsim.Context.Metadata.MESH_TARGET_INST_ID.get());
////        setMetadata(metadata, CONTEXTKEY_SESSION_ID, com.be.mesh.client.prsim.Context.Metadata.MESH_SESSION_ID.get());
//        return MetadataUtils.newAttachHeadersInterceptor(metadata).interceptCall(descriptor, options, channel);
//    }
//
//
//
////    @Override
////    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
////        String remoteAddr = call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
////        String[] remoteAddrSplited = remoteAddr.split(":");
////        String remoteIp = remoteAddrSplited[0].replaceAll("\\/", "");
////        Context context = Context.current().withValue(sourceIp, remoteIp)
////                .withValue()
////
////        return Contexts.interceptCall(context, call, headers, next);
////    }
//
//}