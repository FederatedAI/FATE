package org.fedai.osx.broker.service;

import com.google.inject.Singleton;
import org.fedai.osx.broker.constants.ServiceType;

import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.TYPE;

@Target({TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
@Singleton
public @interface Register {
    ServiceType type() default ServiceType.inner;

    String[] uris();

    boolean allowInterUse() default true;
}
