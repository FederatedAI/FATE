package com.osx.api.translator;


import com.osx.api.context.Context;
import org.ppc.ptp.Osx;
//用于转换不同厂商通信时的接收和发总数据，
public interface Translator {
    //服务方转化接收的数据
    Osx.Inbound  translateReceiveInbound(Context context, Osx.Inbound inbound);
    //请求方转化接受到的返回数据
    Osx.Outbound translateReceiveOutbound(Context  context,Osx.Outbound outbound);
    //请求方转化发送的数据
    Osx.Inbound  translateSendInbound(Context  context,Osx.Inbound inbound);
    //服务方转化准备返回的数据
    Osx.Outbound translateSendOutbound(Context  context,Osx.Outbound outbound);
}
