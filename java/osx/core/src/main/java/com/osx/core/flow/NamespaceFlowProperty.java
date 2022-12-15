
package com.osx.core.flow;

import java.util.List;

public class NamespaceFlowProperty<T> {

    private final String namespace;
    private final Property<List<T>> property;
    private final PropertyListener<List<T>> listener;

    public NamespaceFlowProperty(String namespace,
                                 Property<List<T>> property,
                                 PropertyListener<List<T>> listener) {
        this.namespace = namespace;
        this.property = property;
        this.listener = listener;
    }

    public Property<List<T>> getProperty() {
        return property;
    }

    public String getNamespace() {
        return namespace;
    }

    public PropertyListener<List<T>> getListener() {
        return listener;
    }
}
