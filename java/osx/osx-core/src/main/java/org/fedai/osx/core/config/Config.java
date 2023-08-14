package org.fedai.osx.core.config;

import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.FIELD;

@Target({FIELD})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
public @interface Config {

    String pattern() default  "";
//    String defaultValue() default "";
    String confKey();


}
