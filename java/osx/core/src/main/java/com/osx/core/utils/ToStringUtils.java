package com.osx.core.utils;

import com.osx.core.constant.Dict;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageOrBuilder;
import com.google.protobuf.util.JsonFormat;

public class ToStringUtils {

    private static  JsonFormat.Printer protoPrinter = JsonFormat.printer().preservingProtoFieldNames()

            .includingDefaultValueFields()
            .omittingInsignificantWhitespace();

   public  static  String toOneLineString(MessageOrBuilder target){
        if (target != null) {
            try {
             return    protoPrinter.print(target);
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        else {
            return Dict.NULL_WITH_BRACKETS;
        }

        return  null;
    }



}
