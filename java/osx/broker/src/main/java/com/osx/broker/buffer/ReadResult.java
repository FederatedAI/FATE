package com.osx.broker.buffer;

public    class ReadResult{
        public  ReadResult(ReadStatus status,byte[] data,int readIndex){
            this.status = status;
            this.data = data;
            this.readIndex = readIndex;
        }
        ReadStatus   status;

        public ReadStatus getStatus() {
            return status;
        }

        public void setStatus(ReadStatus status) {
            this.status = status;
        }

        public byte[] getData() {
            return data;
        }

        public void setData(byte[] data) {
            this.data = data;
        }

        public int getReadIndex() {
            return readIndex;
        }

        public void setReadIndex(int readIndex) {
            this.readIndex = readIndex;
        }

        byte[]   data;
        int   readIndex;

    }