package com.osx.broker.queue;


import com.osx.core.context.Context;

public interface Consumer<T> {

//    public   void  fireError(Throwable t);
//
//    public   void  destroy();

//    public   long   addConsumeCount(int size);
//
//    public   long  getConsumeOffset();
//
//    public TransferStatus getTransferStatus() ;
//
//    public void setTransferStatus(TransferStatus transferStatus) ;
//
//    public long getCreateTimestamp() ;
//
//    public void setCreateTimestamp(long createTimestamp) ;

   // public MessageWraper consume(Context context, long offset);
   public T consume(Context context, long offset);

}
