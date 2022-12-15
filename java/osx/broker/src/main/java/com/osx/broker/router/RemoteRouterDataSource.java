package com.osx.broker.router;


import com.osx.core.datasource.AutoRefreshDataSource;
import com.osx.core.datasource.Converter;

public class RemoteRouterDataSource<T> extends AutoRefreshDataSource<String,T> {
    public RemoteRouterDataSource(Converter configParser) {
        super(configParser);
    }


    @Override
    public String readSource() throws Exception {
        return null;
    }
}
