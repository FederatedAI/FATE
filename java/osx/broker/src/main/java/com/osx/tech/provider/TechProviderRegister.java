package com.osx.tech.provider;

import com.google.common.base.Preconditions;
import com.osx.core.frame.Lifecycle;
import com.osx.core.provider.TechProvider;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * 厂商选择
 */
public class TechProviderRegister implements Lifecycle {

    ConcurrentMap<String, TechProvider> registerMap = new ConcurrentHashMap<>();

    //public TechProvider select(Pcp.Inbound inbound) {

        public TechProvider select(String  techProviderCode ) {
            Preconditions.checkArgument(techProviderCode != null);

        return this.registerMap.get(techProviderCode);
    }



    public void init() {
        FateTechProvider fateTechProvider = new FateTechProvider();
        fateTechProvider.init();
        this.registerMap.put(fateTechProvider.getProviderId(), fateTechProvider);
    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }


}



