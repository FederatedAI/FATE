/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.osx.core.constant;

import com.osx.core.config.MetaInfo;

public class Dict {
    public static final String PROPERTY_FATE_TECH_PROVIDER="fate.tech.provider";
    public static final String ORIGIN_REQUEST = "origin_request";
    public static final String CASEID = "caseid";
    public static final String SEQNO = "seqno";
    public static final String NONE = "NONE";
    public static final String GET_REMOTE_PARTY_RESULT = "getRemotePartyResult";
    public static final String PORT = "port";
    public static final String HTTP_PORT ="http.port";

    public static final String INSTANCE_ID = "instanceId";
    public static final String HIT_CACHE = "hitCache";


    public static final String REQUEST_SEQNO = "REQUEST_SEQNO";

    public static final String VERSION = "version";
    public static final String GRPC_TYPE = "grpcType";
    public static final String ROUTER_INFO = "routerInfo";
    public static final String RESULT_DATA = "resultData";
    public static final String RETURN_CODE = "returnCode";
    public static final String SOURCE_PARTY_ID = "sourcePartyId";
    public static final String DES_PARTY_ID = "desPartyId";
    public static final String SOURCE_COMPONENT = "sourceComponent";
    public static final String DES_COMPONENT =  "desComponent";

    public static final String DOWN_STREAM_COST = "downstreamCost";
    public static final String DOWN_STREAM_BEGIN = "downstreamBegin";
    public static final String ROUTE_BASIS = "routeBasis";
    public static final String SOURCE_IP = "sourceIp";
    public static final String PROPERTY_SERVING_CORE_POOL_SIZE = "serving.core.pool.size";
    public static final String SERVING_MAX_POOL_ZIE = "serving.max.pool.size";
    public static final String PROPERTY_SERVING_POOL_ALIVE_TIME = "serving.pool.alive.time";
    public static final String PROPERTY_SERVING_POOL_QUEUE_SIZE = "serving.pool.queue.size";

    public static final String CACHE_TYPE_REDIS = "redis";
    public static final String DEFAULT_FATE_ROOT = "FATE-SERVICES";


    /**
     * configuration property key
     */
    public static final String PROPERTY_SELF_PARTY = "self.party";

    public static final String PROPERTY_CACHE_TYPE = "cache.type";

    public static final String PROPERTY_REDIS_EXPIRE = "redis.expire";
    public static final String PROPERTY_REDIS_CLUSTER_NODES = "redis.cluster.nodes";
    public static final String PROPERTY_LOCAL_CACHE_MAXSIZE = "local.cache.maxsize";
    public static final String PROPERTY_LOCAL_CACHE_EXPIRE = "local.cache.expire";
    public static final String PROPERTY_LOCAL_CACHE_INTERVAL = "local.cache.interval";

    public static final String PROPERTY_GRPC_TIMEOUT = "grpc.timeout";
    public static final String PROPERTY_EXTERNAL_INFERENCE_RESULT_CACHE_DB_INDEX = "external.inferenceResultCacheDBIndex";
    public static final String PROPERTY_EXTERNAL_INFERENCE_RESULT_CACHE_TTL = "external.inferenceResultCacheTTL";
    public static final String PROPERTY_EXTERNAL_REMOTE_MODEL_INFERENCE_RESULT_CACHE_DB_INDEX = "external.remoteModelInferenceResultCacheDBIndex";
    public static final String PROPERTY_EXTERNAL_PROCESS_CACHE_DB_INDEX = "external.processCacheDBIndex";
    public static final String PROPERTY_EXTERNAL_REMOTE_MODEL_INFERENCE_RESULT_CACHE_TTL = "external.remoteModelInferenceResultCacheTTL";
    public static final String PROPERTY_CAN_CACHE_RET_CODE = "canCacheRetcode";
    public static final String PROPERTY_SERVICE_ROLE_NAME = "serviceRoleName";
    public static final String PROPERTY_SERVICE_ROLE_NAME_DEFAULT_VALUE = "serving";
    public static final String PROPERTY_ONLINE_DATA_ACCESS_ADAPTER = "OnlineDataAccessAdapter";
    public static final String PROPERTY_ONLINE_DATA_BATCH_ACCESS_ADAPTER = "OnlineDataBatchAccessAdapter";
    public static final String PROPERTY_MODEL_CACHE_ACCESS_TTL = "modelCacheAccessTTL";
    public static final String PROPERTY_MODEL_CACHE_MAX_SIZE = "modelCacheMaxSize";
    public static final String PROPERTY_INFERENCE_WORKER_THREAD_NUM = "inferenceWorkerThreadNum";
    public static final String PROPERTY_PROXY_ADDRESS = "proxy";
    public static final String ONLINE_ENVIRONMENT = "online";
    public static final String PROPERTY_ROLL_ADDRESS = "roll";
    public static final String PROPERTY_FLOW_ADDRESS = "flow";
    public static final String PROPERTY_SERVING_ADDRESS = "serving";
    public static final String PROPERTY_USE_ZOOKEEPER = "useZookeeper";
    public static final String PROPERTY_PORT = "port";
    public static final String PROPERTY_GRPC_PORT = "grpc.port";
    public static final String PROPERTY_GRPC_TLS_PORT = "grpc.tls.port";
    public static final String PROPERTY_USER_DIR = "user.dir";
    public static final String PROPERTY_USER_HOME = "user.home";
    public static final String PROPERTY_FILE_SEPARATOR = "file.separator";
    public static final String PROPERTY_ZK_URL = "zk.url";
    public static final String PROPERTY_USE_ZK_ROUTER = "useZkRouter";
    public static final String PROPERTY_USE_REGISTER = "useRegister";
    public static final String PROPERTY_MODEL_TRANSFER_URL = "model.transfer.url";
    public static final String PROPERTY_MODEL_SYNC = "model.synchronize";
    public static final String PROPERTY_TRANSFER_FILE_PATH = "transfer.file.path";

    public static final String PROPERTY_FEATURE_BATCH_ADAPTOR = "feature.batch.adaptor";
    public static final String PROPERTY_ACL_ENABLE = "acl.enable";
    public static final String PROPERTY_ACL_USERNAME = "acl.username";
    public static final String PROPERTY_ACL_PASSWORD = "acl.password";
    public static final String PROXY_ROUTER_TABLE = "proxy.router.table";
    public static final String PROPERTY_BATCH_INFERENCE_MAX = "batch.inference.max";
    public static final String PROPERTY_PRINT_INPUT_DATA = "print.input.data";
    public static final String PROPERTY_PRINT_OUTPUT_DATA = "print.output.data";
    public static final String PROPERTY_NEGOTIATIONTYPE = "server.negotiationType";
    public static final String PROPERTY_SERVER_CA_FILE = "server.CA.file";
    public static final String PROPERTY_PROXY_GRPC_INTER_CLIENT_CERTCHAIN_FILE = "client.certChain.file";
    public static final String PROPERTY_PROXY_GRPC_INTER_CLIENT_PRIVATEKEY_FILE = "client.privateKey.file";
    public static final String PROPERTY_SERVER_CERTCHAIN_FILE = "server.certChain.file";
    public static final String PROPERTY_SERVER_PRIVATEKEY_FILE = "server.privateKey.file";
    public static final String CURRENT_VERSION = "currentVersion";

    public static final String PROPERTY_COORDINATOR = "coordinator";
//    public static final String PROPERTY_SERVER_PORT = "server.port";


    public static final String PROPERTY_INFERENCE_SERVICE_NAME = "inference.service.name";
    public static final String PROPERTY_ROUTE_TYPE = "routeType";
    public static final String PROPERTY_ROUTE_TABLE = "route.table";
    public static final String PROPERTY_FLOW_RULE_TABLE = "flow.rule";
    public static final String PROPERTY_AUTH_FILE = "auth.file";
    public static final String PROPERTY_AUTH_OPEN = "auth.open";
    public static final String PROPERTY_PROXY_GRPC_INTRA_PORT = "proxy.grpc.intra.port";
    public static final String PROPERTY_PROXY_GRPC_INTER_PORT = "proxy.grpc.inter.port";
    public static final String PROPERTY_PROXY_GRPC_INFERENCE_TIMEOUT = "proxy.grpc.inference.timeout";
    public static final String PROPERTY_PROXY_GRPC_INFERENCE_ASYNC_TIMEOUT = "proxy.grpc.inference.async.timeout";
    public static final String PROPERTY_PROXY_GRPC_UNARYCALL_TIMEOUT = "proxy.grpc.unaryCall.timeout";
    public static final String PROPERTY_PROXY_GRPC_THREADPOOL_CORESIZE = "proxy.grpc.threadpool.coresize";
    public static final String PROPERTY_PROXY_GRPC_THREADPOOL_MAXSIZE = "proxy.grpc.threadpool.maxsize";
    public static final String PROPERTY_PROXY_GRPC_THREADPOOL_QUEUESIZE = "proxy.grpc.threadpool.queuesize";
    public static final String PROPERTY_PROXY_ASYNC_TIMEOUT = "proxy.async.timeout";
    public static final String PROPERTY_PROXY_ASYNC_CORESIZE = "proxy.async.coresize";
    public static final String PROPERTY_PROXY_ASYNC_MAXSIZE = "proxy.async.maxsize";
    public static final String PROPERTY_PROXY_GRPC_BATCH_INFERENCE_TIMEOUT = "proxy.grpc.batch.inference.timeout";
    public static final String PROPERTY_MODEL_CACHE_PATH = "model.cache.path";
    public static final String PROPERTY_LR_USE_PARALLEL = "lr.use.parallel";
    public static final String PROPERTY_ALLOW_HEALTH_CHECK = "health.check.allow";
    public static final String PROPERTY_TRANSFER_FILE_CACHE_SIZE = "transfer.file.cache.size";
    public static final String PROPERTY_MAX_TRANSFER_CACHE_SIZE = "max.transfer.cache.size";
    public static final String PROPERTY_USE_DIRECT_CACHE = "use.direct.cache";
    public static final String PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT = "grpc.oncompleted.wait.timeout";
//    public static final String PROPERTY_USE_QUEUE_MODEL = "use.queue.model";
    public static final String PROPERTY_STREAM_LIMIT_MODE = "stream.limit.mode";
    public static final String PROPERTY_STREAM_LIMIT_MAX_TRY_TIME = "stream.limit.max.try.time";
    public static final String PROPERTY_GRPC_SERVER_MAX_CONCURRENT_CALL_PER_CONNECTION = "grpc.server.max.concurrent.call.per.connection";
    public static final String PROPERTY_GRPC_SERVER_MAX_INBOUND_MESSAGE_SIZE = "grpc.server.max.inbound.message.size";
    public static final String PROPERTY_GRPC_SERVER_MAX_INBOUND_METADATA_SIZE = "grpc.server.max.inbound.metadata.size";
    public static final String PROPERTY_GRPC_SERVER_FLOW_CONTROL_WINDOW = "grpc.server.flow.control.window";
    public static final String PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC = "grpc.server.keepalive.time.sec";
    public static final String PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC = "grpc.server.keepalive.timeout.sec";
    public static final String PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC = "grpc.server.permit.keepalive.time.sec";
    public static final String PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED = "grpc.server.keepalive.without.calls.enabled";
    public static final String PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC = "grpc.server.max.connection.idle.sec";
    public static final String PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC = "grpc.server.max.connection.age.sec";
    public static final String PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC = "grpc.server.max.connection.age.grace.sec";
    public static final String PROPERTY_INTERVAL_MS = "interval.ms";
    public static final String PROPERTY_SAMPLE_COUNT = "sample.count";
    public static final String PRPPERTY_QUEUE_MAX_FREE_TIME = "queue.max.free.time";

    public static String PROPERTY_OPEN_HTTP_SERVER = "open.http.server";
    public static String PROPERTY_OPEN_GRPC_TLS_SERVER = "open.grpc.tls.server";
    public static String PROPERTY_DEFAULT_CLIENT_VERSION="default.client.version";


    public static final String HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT = "httpclinet.config.connection.req.timeout";
    public static final String HTTP_CLIENT_CONFIG_CONN_TIME_OUT = "httpclient.config.connection.timeout";
    public static final String HTTP_CLIENT_CONFIG_SOCK_TIME_OUT = "httpclient.config.sockect.timeout";
    public static final String HTTP_CLIENT_INIT_POOL_MAX_TOTAL = "httpclient.init.pool.maxtotal";
    public static final String HTTP_CLIENT_INIT_POOL_DEF_MAX_PER_ROUTE = "httpclient.init.pool.def.max.pre.route";
    public static final String HTTP_CLIENT_INIT_POOL_SOCK_TIME_OUT = "httpclient.init.pool.sockect.timeout";
    public static final String HTTP_CLIENT_INIT_POOL_CONN_TIME_OUT = "httpclient.init.pool.connection.timeout";
    public static final String HTTP_CLIENT_INIT_POOL_CONN_REQ_TIME_OUT = "httpclient.init.pool.connection.req.timeout";
    public static final String HTTP_CLIENT_TRAN_CONN_REQ_TIME_OUT = "httpclient.tran.connection.req.timeout";
    public static final String HTTP_CLIENT_TRAN_CONN_TIME_OUT = "httpclient.tran.connection.timeout";
    public static final String HTTP_CLIENT_TRAN_SOCK_TIME_OUT = "httpclient.tran.sockect.timeout";


    public static final String ACTION_TYPE_ASYNC_EXECUTE = "ASYNC_EXECUTE";

    public static final String RET_CODE = "retcode";
    public static final String RET_MSG = "retmsg";
    public static final String DATA = "data";
    public static final String STATUS = "status";
    public static final String SUCCESS = "success";
    public static final String PROB = "prob";
    public static final String ACCESS = "access";

    public static final String TAG_INPUT_FORMAT = "tag";
    public static final String SPARSE_INPUT_FORMAT = "sparse";
    public static final String MIN_MAX_SCALE = "min_max_scale";
    public static final String STANDARD_SCALE = "standard_scale";

    public static final String HOST = "host";
    public static final String GUEST = "guest";
    public static final String PARTNER_PARTY_NAME = "partnerPartyName";
    public static final String PARTY_NAME = "partyName";

    public static final String FEDERATED_PARAMS = "federatedParams";
    public static final String COMMIT_ID = "commitId";
    public static final String BRANCH_MASTER = "master";

    public static final String SERVICE_NAME = "serviceName";
    public static final String CALL_NAME = "callName";
    public static final String INPUT_DATA = "input_data";
    public static final String OUTPUT_DATA = "output_data";
    public static final String PROPERTY_SERVING_MAX_POOL_SIZE = "serving.max.pool.size";

    public static final String CONTENT_TYPE = "Content-Type";
    public static final String CONTENT_TYPE_JSON_UTF8 = "application/json;charset=UTF-8";
    public static final String CHARSET_UTF8 = "UTF-8";
    public static final String HTTP = "http";
    public static final String HTTPS = "https";
    public static final String UNARYCALL = "unaryCall";

    public static final String GUEST_APP_ID = "guestAppId";
    public static final String HOST_APP_ID = "hostAppId";
    public static final String SERVICE_ID = "serviceId";
    public static final String APPLY_ID = "applyId";
    public static final String FUTURE = "future";
    public static final String AUTH_FILE = "authFile";
    public static final String ENCRYPT_TYPE = "encrypt_type";


    public static final String CASE_ID = "caseid";
    public static final String CODE = "code";
    public static final String MESSAGE = "message";
    public static final String MODEL_ID = "modelId";
    public static final String MODEL_VERSION = "modelVersion";
    public static final String TIMESTAMP = "timestamp";
    public static final String APP_ID = "appid";
    public static final String PARTY_ID = "partyId";
    public static final String ROLE = "role";
    public static final String PART_ID = "partId";
    public static final String FEATURE_DATA = "featureData";
    public static final String SESSION_TOKEN = "sessionToken";

    public static final String DEFAULT_VERSION = "1.0";
    public static final String SELF_PROJECT_NAME = "proxy";
    public static final String SELF_ENVIRONMENT = "online";
    public static final String HEAD = "head";
    public static final String BODY = "body";


    public static final String SBT_TREE_NODE_ID_ARRAY = "sbtTreeNodeIdArray";

    public static final String REMOTE_METHOD_BATCH = "batch";
    public static final String MODEL_NAME_SPACE = "modelNameSpace";
    public static final String MODEL_TABLE_NAME = "modelTableName";
    public static final String REGISTER_ENVIRONMENT = "online";

    public static final String SERVICE_FIREWORK = "firework";
    public static final String SERVICE_FIREWORK_CLUSTERMANAGER = "firework_cluster_manager";
    public static final String SERVICE_PROXY = "proxy";
    public static final String SERVICE_ADMIN = "admin";
    public static final String FAILED = "failed";
    public static final String BATCH_PRC_TIMEOUT = "batch.rpc.timeout";
    public static final String PASS_QPS = "passQps";
    // parameters
    public static final String PARAMS_INITIATOR = "initiator";
    public static final String PARAMS_ROLE = "role";
    public static final String PARAMS_JOB_PARAMETERS = "job_parameters";
    public static final String PARAMS_SERVICE_ID = "service_id";
    public static final String BATCH_INFERENCE_SPLIT_SIZE = "batch.inference.split.size";
    public static final String WARN_LIST = "warnList";
    public static final String ERROR_LIST = "errorList";
    public static final String HEALTH_INFO = "healthInfo";
    public static final String PROPERTY_ADMIN_HEALTH_CHECK_TIME = "health.check.time";


    public static final String ROLLSITE_ROUTE_TABLE_KEY = "rollsite.route.table.key";
    public static final String ROLLSITE_ROUTE_TABLE_WHITE_LIST = "rollsite.route.table.whitList";
    public static final String ROLLSITE_ROUTE_TABLE_PARTY_ID = "rollsite.route.table.party.id";

    public static final String ROLLSITE_PARTY_ID = "rollsite.party.id";

    public static final String TRANSFER_FATECLOUD_AHTHENTICATION_ENABLED = "transfer.fateCloud.authentication.enabled";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_USE_CONFIG = "transfer.fateCloud.authentication.use.config";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_URI = "transfer.fateCloud.authentication.uri";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_APPKEY = "transfer.fateCloud.authentication.appkey";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_APPSERCRET = "transfer.fateCloud.authentication.appsecret";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_ROLE = "transfer.fateCloud.authentication.role";
    public static final String TRANSFER_FATECLOUD_SECRET_INFO_URL = "transfer.fateCloud.secret.info.url";
    public static final String TRANSFER_FATECLOUD_AUTHENTICATION_URL = "transfer.fateCloud.authentication.url";
    public static final String NETTY_CHANNEL = "netty.channel";
    public static final String RESPONSE_STREAM_OBSERVER = "response.stream.observer";

    public static final String REQUEST_INDEX = "request.index";
    public static final String CURRENT_INDEX = "current.index";
    public static final String TRANSFER_ID = "transferId";
    public static final String TOPIC = "topic";


    public static final String PROPERTY_DEPLOY_MODE = "deploy.model";
    public static final String PROPERTY_CLUSTER_MANAGER_ADDRESS = "cluster.manager.address";


    public static final String PROPERTY_EGGROLL_CLUSTER_MANANGER_IP = "eggroll.cluster.manager.ip";
    public static final String PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT = "eggroll.cluster.manager.port";
    public final static String UNKNOWN = "UNKNOWN";
    public final static String PROTOBUF = "PROTOBUF";
    public final static String SLASH = "/";
    public final static String GRPC_PARSE_FROM = "parseFrom";
    public final static String AT = "@";
    public final static String AND = "&";
    public final static String EQUAL = "=";
    public final static String DOLLAR = "$";
    public final static String DOT = ".";
    public final static String COLON = ":";
    public final static String SEMICOLON = ";";
    public final static String DASH = "-";
    public final static String UNDERLINE = "_";
    public final static String DOUBLE_UNDERLINES = "__";
    public final static String COMMA = ",";
    public final static String HASH = "#";
    public final static String META = "meta";
    public final static String SEND_START = "send_start";

//        public final static String HOST = "host";
//        public final static String PORT = "port";
    public final static String SEND_END = "send_end";
    public final static String DEFAULT = "default";
    public final static String ROLE_EGG = "egg";
    public final static String ROLE_ROLL = "roll";
    public final static String ROLE_EGGROLL = "eggroll";
    public final static String EGGROLL_COMPATIBLE_ENABLED = "eggroll.compatible.enabled";
    public final static String FALSE = "false";
    public final static String TRUE = "true";
    public final static String CLUSTER_COMM = "__clustercomm__";
    public final static String FEDERATION = "__federation__";
    public final static String EGGROLL = "eggroll";
    public final static String COMPUTING = "computing";
    public final static String STORAGE = "storage";
    public final static String EMPTY = "";
    public final static String SPACE = " ";
    public final static String LOGGING_A_THROWABLE = "logging a Throwable";
    public final static String ROUTE = "route";
    public final static String NULL = "null";
    public final static String NULL_WITH_BRACKETS = "[${NULL}]";
    public final static String LF = "\n";
    public final static String LFLF = "\n\n";
    public final static String PATH = "path";
    public final static String TYPE = "type";
    public final static String SIZE = "size";
    public final static String ROLL_PAIR = "rollpair";
    public final static String ROLL_FRAME = "rollframe";
    public final static String LMDB = "lmdb";
    public final static String LEVELDB = "leveldb";
    public final static String FILE = "file";
    public final static String HDFS = "hdfs";
    public final static String NETWORK = "network";
    public final static String CACHE = "cache";
    public final static String QUEUE = "queue";
    public final static String TOTAL = "total";
    public final static String LOCALHOST = "localhost";
    public final static String STORE_TYPE = "storeType";
    public final static String STORE_TYPE_SNAKECASE = "store_type";
    public final static String NAMESPACE = "namespace";
    public final static String NAME = "name";
    public final static String TOTAL_PARTITIONS = "totalPartitions";
    public final static String TOTAL_PARTITIONS_SNAKECASE = "total_partitions";
    public final static String PARTITION_ID_SNAKECASE = "partition_id";
    public final static String PARTITIONER = "partitioner";
    public final static String SERDES = "serdes";
    public final static String TRANSFER_BROKER_NAME = "transfer_broker_name";
    public static String PROPERTY_DLEDGER_PEER = "dledger.peer";
    public static String PROPERTY_DLEDGER_SELF = "dledger.self";


}
