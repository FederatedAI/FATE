package com.osx.core.constant;

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
    REDIRECT_ACK("redirect-ack"),
    UNARY_CALL("unary-call"),
    UNARY_CALL_NEW("unary-call-new");


//    "redirect-ack"


    private   String alias;

    private  ActionType(String alias){
        this.alias = alias;
    }
    public  String getAlias(){
        return  alias;
    }


}
