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

package org.fedai.osx.core.constant;

public class Dict {

    public static final String ORIGIN_REQUEST = "origin_request";
    public static final String CASEID = "caseid";
    public static final String SEQNO = "seqno";
    public static final String NONE = "NONE";
    public static final String GET_REMOTE_PARTY_RESULT = "getRemotePartyResult";
    public static final String PORT = "port";
    public static final String HTTP_PORT ="http.port";

    public static final String INSTANCE_ID = "instanceId";
    public static final String POSITIVE_INTEGER_PATTERN = "^[1-9]\\d*$";
    public static final String BOOLEAN_PATTERN="^(true)|(false)$";
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
    //HttpServletResponse
    public static final String HTTP_SERVLET_RESPONSE = "httpServletResponse";
    public static final String HTTP_ASYNC_CONTEXT = "AsyncContext";




//    public static final String PROPERTY_BIND_HOST_KEY = "bind.host";

    /**
     * configuration property key
     */
    public static final String PROPERTY_SELF_PARTY_KEY = "self.party";
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






    public static final String ACTION_TYPE_ASYNC_EXECUTE = "ASYNC_EXECUTE";

    public static final String RET_CODE = "retcode";
    public static final String RET_MSG = "retmsg";
    public static final String DATA = "data";
    public static final String STATUS = "status";
    public static final String SUCCESS = "success";
    public static final String DUP_MSG = "dup_msg";
    public static final String PROCESSED_MSG = "Processed messages";
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
    public static final String MESSAGE_FLAG = "message_flag";
    public static final String MESSAGE_CODE = "message_code";
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
    public static final String SESSION_ID = "sessionId";
    public static final String METHOD_CONFIG_REQ_TIMEOUT = "reqTimeout";
    public static final String METHOD_CONFIG_CONNECTION_TIMEOUT = "connectionTimeout";
    public static final String METHOD_CONFIG_SOCKET_TIMEOUT = "socketTimeout";

    public static final String SBT_TREE_NODE_ID_ARRAY = "sbtTreeNodeIdArray";

    public static final String REMOTE_METHOD_BATCH = "batch";
    public static final String MODEL_NAME_SPACE = "modelNameSpace";
    public static final String MODEL_TABLE_NAME = "modelTableName";
    public static final String REGISTER_ENVIRONMENT = "online";

    public static final String SERVICE_OSX = "osx";
    public static final String SERVICE_OSX_CLUSTERMANAGER = "osx_cluster_manager";
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

    public static final String PAYLOAD = "payload";


    public static final String PROPERTY_DEPLOY_MODE_KEY = "deploy.model";
//    public static final String PROPERTY_CLUSTER_MANAGER_ADDRESS = "cluster.manager.address";


    public static final String PROPERTY_EGGROLL_CLUSTER_MANANGER_IP_KEY = "eggroll.cluster.manager.ip";
    public static final String PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT_KEY = "eggroll.cluster.manager.port";
    public final static String UNKNOWN = "UNKNOWN";
    public final static String PROTOBUF = "PROTOBUF";
    public final static String SLASH = "/";
    public final static String COMPONENTS_DIR = "components";
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
    public final static String LOCALHOST2 = "127.0.0.1";
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
    public final static String TRANSFER_QUEUE = "transfer_queue";
    public final static String IS_CYCLE="cycle";
//    public final static String EGGROLL_SEND_TOPIC_PREFIX="EGGROLL_SEND_";
//    public final static String EGGROLL_BACK_TOPIC_PREFIX="EGGROLL_BACK_";
    public final static String STREAM_SEND_TOPIC_PREFIX = "STREAM_SEND_";
    public final static String STREAM_BACK_TOPIC_PREFIX = "STREAM_BACK_";
    public final static String BLOCKING_STUB = "BLOCKING_STUB";
    public final static String PROTOCOL = "protocol";
    public final static String URL="url";

    public final static String USE_SSL="useSSL";
    public final static String CA_FILE="caFile";
    public final static String CERT_CHAIN_FILE="certChainFile";
    public final static String PRIVATE_KEY_FILE="privateKeyFile";

    public final static String KEYSTORE_FILE="keyStoreFile";
    public final static String TRUSTSTORE_FILE="trustStoreFile";
    public final static String KEYSTORE_PASSWORD="keyStorePassword";
    public final static String TRUSTSTORE_PASSWORD="trustStorePassword";



}
