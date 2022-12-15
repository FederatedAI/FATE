
package com.osx.broker.message;

public class MessageSysFlag {
    public final static int COMPRESSED_FLAG = 0x1;
    public final static int MULTI_TAGS_FLAG = 0x1 << 1;
    public final static int TRANSACTION_NOT_TYPE = 0;
    public final static int TRANSACTION_PREPARED_TYPE = 0x1 << 2;
    public final static int TRANSACTION_COMMIT_TYPE = 0x2 << 2;
    public final static int TRANSACTION_ROLLBACK_TYPE = 0x3 << 2;
    public final static int BORNHOST_V6_FLAG = 0x1 << 4;
    public final static int STOREHOSTADDRESS_V6_FLAG = 0x1 << 5;

    public static int getTransactionValue(final int flag) {
        return flag & TRANSACTION_ROLLBACK_TYPE;
    }

    public static int resetTransactionValue(final int flag, final int type) {
        return (flag & (~TRANSACTION_ROLLBACK_TYPE)) | type;
    }

    public static int clearCompressedFlag(final int flag) {
        return flag & (~COMPRESSED_FLAG);
    }

}
