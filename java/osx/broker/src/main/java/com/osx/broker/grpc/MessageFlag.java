package com.osx.broker.grpc;

public enum MessageFlag {

    MSG(0), ERROR(1), COMPELETED(2);

    private int flag;

    private MessageFlag(int flag) {
        this.flag = flag;
    }

    static public MessageFlag getMessageFlag(int flag) {
        switch (flag) {
            case 0:
                return MSG;
            case 1:
                return ERROR;
            case 2:
                return COMPELETED;
            default:
                return null;
        }
    }

    public int getFlag() {
        return flag;
    }


}
