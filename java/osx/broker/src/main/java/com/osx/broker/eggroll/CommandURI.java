package com.osx.broker.eggroll;


import com.osx.core.constant.Dict;
import org.apache.commons.lang3.StringUtils;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class CommandURI {
    URI uri;
    String queryString;
    Map<String,String > queryPairs;

    public CommandURI(String prefix,String name ){
        this(prefix+"/"+name);
    }
    public CommandURI(String  uriString ){
        try {
            uri = new URI(uriString);
            queryString = uri.getQuery();
            queryPairs =  new HashMap<>();
            if (StringUtils.isBlank(queryString)) {
                queryPairs.put(Dict.ROUTE, uriString);
            } else {
                for (String pair : queryString.split(Dict.AND)) {
                    int idx = pair.indexOf(Dict.EQUAL);

                    String key = pair;
                    String value = Dict.EMPTY;
                    if (idx > 0)
                        key = URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8.name()) ;
                    else
                         if (idx > 0 && pair.length() > idx + 1)
                                value = URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8.name());
                    queryPairs.put(key, value);
                }
            }

        } catch (Exception  e) {
            e.printStackTrace();
        }

    }



    String getQueryValue(String  key){
        return    queryPairs.get(key);
    }

    String getRoute(){
       return  queryPairs.get(Dict.ROUTE);
    }


    String  getName() {
      return   StringUtils.substringAfterLast(uri.getPath(), Dict.SLASH);
    }



//    class CommandURI(val uriString: String) {
//        val uri = new URI(uriString)
//        val queryString = uri.getQuery
//        private val queryPairs = mutable.Map[String, String]()
//
//        def this(src: ErCommandRequest) {
//            this(src.uri)
//        }
//
//        def this(prefix: String, name: String) {
//            this(s"${prefix}/${name}")
//        }
//
//        def getName(): String = {
//            StringUtils.substringAfterLast(uri.getPath, StringConstants.SLASH)
//        }

  /*  def this(src: ErCommandResponse) {
      this(src.request.uri)
    }*/

//  if (StringUtils.isBlank(queryString)) {
//            queryPairs.put(StringConstants.ROUTE, uriString)
//        } else {
//            for (pair <- queryString.split(StringConstants.AND)) {
//                val idx = pair.indexOf(StringConstants.EQUAL)
//                val key = if (idx > 0) URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8.name()) else pair
//                val value = if (idx > 0 && pair.length > idx + 1) URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8.name()) else StringConstants.EMPTY
//                queryPairs.put(key, value)
//            }
//        }
//
//        def getQueryValue(key: String): String = {
//            queryPairs(key)
//        }
//
//        def getRoute(): String = {
//            queryPairs(StringConstants.ROUTE)
//        }
//    }
}
