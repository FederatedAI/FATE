package com.osx.core.flow;

public abstract class AbstractRule implements Rule {

    /**
     * Resource name.
     */
    private String resource;

    /**
     * <p>
     * Application name that will be limited by origin.
     * The default limitApp is {@code default}, which means allowing all origin apps.
     * </p>
     * <p>
     * For authority rules, multiple origin name can be separated with comma (',').
     * </p>
     */
    private String limitApp;

    @Override
    public String getResource() {
        return resource;
    }

    public AbstractRule setResource(String resource) {
        this.resource = resource;
        return this;
    }

    public String getLimitApp() {
        return limitApp;
    }

    public AbstractRule setLimitApp(String limitApp) {
        this.limitApp = limitApp;
        return this;
    }

//    @Override
//    public boolean equals(Object o) {
//        if (this == o) {
//            return true;
//        }
//        if (!(o instanceof AbstractRule)) {
//            return false;
//        }
//
//        AbstractRule that = (AbstractRule)o;
//
//        if (resource != null ? !resource.equals(that.resource) : that.resource != null) {
//            return false;
//        }
//        if (!limitAppEquals(limitApp, that.limitApp)) {
//            return false;
//        }
//        return true;
//    }

//    private boolean limitAppEquals(String str1, String str2) {
//        if ("".equals(str1)) {
//            return RuleConstant.LIMIT_APP_DEFAULT.equals(str2);
//        } else if (RuleConstant.LIMIT_APP_DEFAULT.equals(str1)) {
//            return "".equals(str2) || str2 == null || str1.equals(str2);
//        }
//        if (str1 == null) {
//            return str2 == null || RuleConstant.LIMIT_APP_DEFAULT.equals(str2);
//        }
//        return str1.equals(str2);
//    }

    public <T extends AbstractRule> T as(Class<T> clazz) {
        return (T) this;
    }

//    @Override
//    public int hashCode() {
//        int result = resource != null ? resource.hashCode() : 0;
//        if (!("".equals(limitApp) || RuleConstant.LIMIT_APP_DEFAULT.equals(limitApp) || limitApp == null)) {
//            result = 31 * result + limitApp.hashCode();
//        }
//        return result;
//    }
}
