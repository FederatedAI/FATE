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

public enum ActionType {

    QUERY_TOPIC("query-topic"),
    CANCEL_TOPIC("cancel-topic"),
    PUSH_REMOTE("push-to-remote"),
    PUSH_EGGROLL("push-to-eggroll"),
    LOCAL_ACK("local-ack"),
    CUSTOMER_CONSUME("customer-consume"),
    DEFUALT_CONSUME("default-consume"),
    REDIRECT_CONSUME("redirect-consume"),
    INNER_REDIRECT("inner-redirect"),
    LONG_PULLING_ANSWER("long-pulling-answer"),
    MSG_DOWNLOAD("msg-download"),
    MSG_REDIRECT("msg-redirect"),
    REDIRECT_ACK("redirect-ack"),
    UNARY_CALL("unary-call"),
    UNARY_CALL_NEW("unary-call-new"),
    CLUSTER_TOKEN_APPLY("cluster-token-apply"),
    TOPIC_APPLY("topic_apply");

    private   String alias;
    private  ActionType(String alias){
        this.alias = alias;
    }
    public  String getAlias(){
        return  alias;
    }


}
