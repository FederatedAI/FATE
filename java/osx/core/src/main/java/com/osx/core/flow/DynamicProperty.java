
package com.osx.core.flow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;



public class DynamicProperty<T> implements Property<T> {
    Logger logger = LoggerFactory.getLogger(DynamicProperty.class);

    protected Set<PropertyListener<T>> listeners = Collections.synchronizedSet(new HashSet<PropertyListener<T>>());
    private T value = null;

    public DynamicProperty() {
    }

    public DynamicProperty(T value) {
        super();
        this.value = value;
    }

    @Override
    public void addListener(PropertyListener<T> listener) {

        listeners.add(listener);
       // logger.info("fire listener config load {}",value);
        listener.configLoad(value);
    }

    @Override
    public void removeListener(PropertyListener<T> listener) {
        listeners.remove(listener);
    }

    @Override
    public boolean updateValue(T newValue) {
        if (isEqual(value, newValue)) {
            return false;
        }
      //  RecordLog.info("[DynamicSentinelProperty] Config will be updated to: {}", newValue);

        value = newValue;
        for (PropertyListener<T> listener : listeners) {
            listener.configUpdate(newValue);
        }
        return true;
    }

    private boolean isEqual(T oldValue, T newValue) {
        if (oldValue == null && newValue == null) {
            return true;
        }

        if (oldValue == null) {
            return false;
        }

        return oldValue.equals(newValue);
    }

    public void close() {
        listeners.clear();
    }
}
