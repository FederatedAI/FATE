package org.fedai.osx.core.utils;

import org.apache.commons.lang3.StringUtils;

public class UrlUtil {

    public static String parseUri(String oriUri) {
        String result = oriUri;
        if (StringUtils.isNotEmpty(oriUri) && oriUri.contains("://")) {
            String[] args = oriUri.split("://");
            if (args.length > 1) {
                int index = args[1].indexOf("/");
                if (index > 1) {
                    result = args[1].substring(index);
                }
            }
        }
        return result;
    }

    public static String buildUrl(String protocol, String host, String uri) {
        if (StringUtils.isNotEmpty(uri) && !uri.contains("://")) {
            StringBuilder sb = new StringBuilder();
            return sb.append(protocol).append(host).append(uri).toString();
        } else {
            return uri;
        }
    }
}
