package com.osx.broker.queue;

import com.osx.core.utils.JsonUtil;

public   class CreateQueueResult{
        TransferQueue  transferQueue;
        String  redirectIp;
        int  port;
        public TransferQueue getTransferQueue() {
            return transferQueue;
        }

        public void setTransferQueue(TransferQueue transferQueue) {
            this.transferQueue = transferQueue;
        }

        public String getRedirectIp() {
            return redirectIp;
        }

        public void setRedirectIp(String redirectIp) {
            this.redirectIp = redirectIp;
        }

        public int getPort() {
            return port;
        }

        public void setPort(int port) {
            this.port = port;
        }

        public  String toString(){
            return JsonUtil.object2Json(this);
        }

    }