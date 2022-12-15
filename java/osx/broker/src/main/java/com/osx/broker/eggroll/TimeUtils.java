package com.osx.broker.eggroll;

import org.apache.commons.lang3.StringUtils;

import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TimeUtils {
    static DateTimeFormatter noSeparatorFormatter = DateTimeFormatter.ofPattern("yyyyMMdd.HHmmss.SSS");
        public static  String getNowMs(String dateFormat) {
        LocalDateTime now = LocalDateTime.now();
        if (StringUtils.isBlank(dateFormat)) {
          return  noSeparatorFormatter.format(now);
        } else {
            return  new SimpleDateFormat(dateFormat).format(now);
        }
        }
    }

