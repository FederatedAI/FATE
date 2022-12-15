//package com.firework.transfer.grpc;
//
//import com.firework.transfer.message.AllocateMappedFileService;
//import com.firework.transfer.message.Message;
//import com.firework.transfer.message.MessageExtBrokerInner;
//import com.firework.transfer.queue.PutMessageResult;
//import com.firework.transfer.queue.TransferQueue;
//import com.firework.transfer.queue.TransferQueueManager;
//
//import java.net.InetSocketAddress;
//
//public class Test {
//
//
//
//    public static void main(String[] args){
//
//        AllocateMappedFileService allocateMappedFileService = new AllocateMappedFileService();
//        allocateMappedFileService.start();
//        TransferQueueManager  transferQueueManager = new TransferQueueManager(allocateMappedFileService);
//        TransferQueue transferQueue = transferQueueManager.createNewQueue("kaidengTest");
//        for(int i=0;i<10;i++) {
//            MessageExtBrokerInner messageExtBrokerInner = new MessageExtBrokerInner();
//            messageExtBrokerInner.setBody(("kaideng"+i).getBytes());
//            messageExtBrokerInner.setTopic("kaidengTopic");
//            messageExtBrokerInner.setBornHost(new InetSocketAddress(1111));
//            PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
////            System.err.println(putMessageResult);
////            SelectMappedBufferResult selectMappedBufferResult = transferQueue.selectOneMessageByOffset(0);
////            System.err.println(selectMappedBufferResult.getStartOffset() + " " + selectMappedBufferResult.getSize());
//////        selectMappedBufferResult.getByteBuffer()
////
////            MessageExt messageExt = MessageDecoder.decode(selectMappedBufferResult.getByteBuffer());
////            byte[] body = messageExt.getBody();
////            System.err.println(messageExt.getStoreSize());
////            System.err.println(new String(body));
//        }
//
//        for(int i=0;i<10;i++){
//            Message message = null;
////            try {
////                message = transferQueue.consumeOneMessage();
////            } catch (InterruptedException e) {
////                e.printStackTrace();
////            }
//            System.err.println(new String(message.getBody()));
//        }
//
//        transferQueue.destory();
//    }
////    public  static void  main(String[] args){
////
////        int  bufferSize = 1024;
////        ByteBuffer writeBuffer = ByteBuffer.allocateDirect(bufferSize);
////        ByteBuffer  readBuffer = writeBuffer.slice();
////
////        BlockingQueue<Integer>  blockingQueue = new ArrayBlockingQueue(1000);
////
////        new  Thread(()->{
////                for(int i=0;i<10;i++){
////
////                    byte[] bytes = ("kaideng"+i).getBytes(StandardCharsets.UTF_8);
////                    int  remaining = writeBuffer.remaining();
////                    int  writePosition = writeBuffer.position();
////                    int  readPosition = readBuffer.position();
////                   if( bytes.length>remaining){
////                       if(writePosition+bytes.length-bufferSize<readPosition){
////                           System.err.println("==========================");
////                           byte[] part1= Arrays.copyOfRange(bytes,0,remaining-1);
////                           byte[] part2= Arrays.copyOfRange(bytes,remaining,bytes.length-1);
////                           writeBuffer.put(part1);
////                           writeBuffer.position(0);
////                           writeBuffer.put(part2);
////                           blockingQueue.add(bytes.length);
////                       }else{
////                        //追上了读
////                           System.err.println("write over read "+(writePosition+bytes.length-bufferSize)+" "+readPosition);
////                       }
////                   }else {
////                       blockingQueue.add(bytes.length);
////                       writeBuffer.put(bytes);
////                   }
////
////                }
////        }).start();
////
////        new  Thread(()->{
////            while(true){
////                try {
////                 Integer size = blockingQueue.take();
////                 int  writePosition = writeBuffer.position();
////                 int  readPosition = readBuffer.position();
////                  int  remaining =   readBuffer.remaining();
////
////                    byte[] data;
////                 if(size>remaining){
////
////                     byte[] part1= new byte[remaining];
////                     byte[] part2 =  new  byte[size-remaining];
////                     readBuffer.get(part1);
////                     readBuffer.position(0);
////                     readBuffer.get(part2);
////
////                    data = concat(part1,part2);
////
////
////                 }else
////                    {
////
////                        data = new byte[size];
////                        readBuffer.get(data);
////                    }
////
////                      writePosition = writeBuffer.position();
////                      readPosition = readBuffer.position();
////
////
////                 System.err.println(new String(data));
////                } catch (InterruptedException e) {
////                    e.printStackTrace();
////                }
////
////            }
////        }).start();
//
//
////        System.err.println(writeBuffer.position() +"   "+writeBuffer.limit()+" "+writeBuffer.capacity());
////        System.err.println(readBuffer.position() +"   "+readBuffer.limit()+" "+readBuffer.capacity());
////        byte[] bytes = "kaideng".getBytes(StandardCharsets.UTF_8);
////        writeBuffer.put(bytes);
////        System.err.println(writeBuffer.position() +"   "+writeBuffer.limit()+" "+writeBuffer.capacity());
////        System.err.println(readBuffer.position() +"   "+readBuffer.limit()+" "+readBuffer.capacity());
////        int length = bytes.length;
////
////        byte[] newBytes = new  byte[length];
////
//////        writeBuffer.flip();
//////        writeBuffer.position(0);
////        readBuffer.get(newBytes,0,length);
////
////        System.err.println(new String(newBytes));
////        System.err.println(writeBuffer.position() +"   "+writeBuffer.limit()+" "+writeBuffer.capacity());
////        System.err.println(readBuffer.position() +"   "+readBuffer.limit()+" "+readBuffer.capacity());
//
//
////    }
//
//    static byte[] concat(byte[] a, byte[] b) {
//        byte[] c= new byte[a.length+b.length];
//        System.arraycopy(a, 0, c, 0, a.length);
//        System.arraycopy(b, 0, c, a.length, b.length);
//        return c;
//    }
//}
