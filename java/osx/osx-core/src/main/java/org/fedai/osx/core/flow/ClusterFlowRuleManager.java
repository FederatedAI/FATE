/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.core.flow;

import com.fasterxml.jackson.core.type.TypeReference;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.datasource.FileRefreshableDataSource;
import org.fedai.osx.core.utils.AssertUtil;
import org.fedai.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public final class ClusterFlowRuleManager {

    public static final Function<String, Property<List<FlowRule>>> DEFAULT_PROPERTY_SUPPLIER =
            new Function<String, Property<List<FlowRule>>>() {
                @Override
                public Property<List<FlowRule>> apply(String namespace) {
                    return new DynamicProperty<>();
                }
            };

    private static final Map<Long, FlowRule> FLOW_RULES = new ConcurrentHashMap<>();

    private static final Map<String, FlowRule> RESOURCE_RULES = new ConcurrentHashMap<>();

    private static final Map<String, Set<Long>> NAMESPACE_FLOW_ID_MAP = new ConcurrentHashMap<>();

    private static final Map<Long, String> FLOW_NAMESPACE_MAP = new ConcurrentHashMap<>();

    private static final Map<String, NamespaceFlowProperty<FlowRule>> PROPERTY_MAP = new ConcurrentHashMap<>();
    private static final Object UPDATE_LOCK = new Object();
    static Logger logger = LoggerFactory.getLogger(ClusterFlowRuleManager.class);
    private static volatile Function<String, Property<List<FlowRule>>> propertySupplier
            = DEFAULT_PROPERTY_SUPPLIER;

    static {
        initDefaultProperty();
    }

    public ClusterFlowRuleManager() {
    }

    private static void initDefaultProperty() {
        logger.info("ClusterFlowRuleManager initDefaultProperty");
        Property<List<FlowRule>> defaultProperty = new DynamicProperty<>();
        String defaultNamespace = "default";
        PropertyListener<List<FlowRule>> listener = new FlowRulePropertyListener(defaultNamespace);
        registerPropertyInternal(defaultNamespace, defaultProperty, listener);
        String currentPath = null;
        //先考虑开发本地情况、不是本地再按服务器方式获取
        currentPath = MetaInfo.PROPERTY_CONFIG_DIR + "/" + MetaInfo.PROPERTY_FLOW_RULE_TABLE;
        logger.info("load flow rule {}", currentPath);
        File confFile = new File(currentPath);
        FileRefreshableDataSource fileRefreshableDataSource;
        try {
            fileRefreshableDataSource = new FileRefreshableDataSource(confFile, (source) -> {

                List<FlowRule> content = JsonUtil.json2List((String) source, new TypeReference<List<FlowRule>>() {
                });
                logger.info("load flow rule content {}", content);
                return content;
            });
            fileRefreshableDataSource.getProperty().addListener(listener);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            logger.error("flow rule file not exist");
        }

    }

    public static void setPropertySupplier(Function<String, Property<List<FlowRule>>> propertySupplier) {
        AssertUtil.notNull(propertySupplier, "flow rule property supplier cannot be null");
        ClusterFlowRuleManager.propertySupplier = propertySupplier;
    }

    /**
     * Listen to the {@link Property} for cluster {@link FlowRule}s.
     * The property is the source of cluster {@link FlowRule}s for a specific namespace.
     *
     * @param namespace namespace to register
     */
    public static void register2Property(String namespace) {
        AssertUtil.notEmpty(namespace, "namespace cannot be empty");
        if (propertySupplier == null) {
//            RecordLog.warn(
//                    "[ClusterFlowRuleManager] Cluster flow property supplier is absent, cannot register property");
            return;
        }
        Property<List<FlowRule>> property = propertySupplier.apply(namespace);
        if (property == null) {
//            RecordLog.warn(
//                    "[ClusterFlowRuleManager] Wrong created property from cluster flow property supplier, ignoring");
            return;
        }
        synchronized (UPDATE_LOCK) {
//            RecordLog.info("[ClusterFlowRuleManager] Registering new property to cluster flow rule manager"
//                    + " for namespace <{}>", namespace);
            registerPropertyInternal(namespace, property);
        }
    }

    public static void registerPropertyIfAbsent(String namespace) {
        AssertUtil.notEmpty(namespace, "namespace cannot be empty");
        if (!PROPERTY_MAP.containsKey(namespace)) {
            synchronized (UPDATE_LOCK) {
                if (!PROPERTY_MAP.containsKey(namespace)) {
                    register2Property(namespace);
                }
            }
        }
    }

    private static void registerPropertyInternal(/*@NonNull*/ String namespace, /*@Valid*/
                                                              Property<List<FlowRule>> property

    ) {
        NamespaceFlowProperty<FlowRule> oldProperty = PROPERTY_MAP.get(namespace);
        if (oldProperty != null) {
            oldProperty.getProperty().removeListener(oldProperty.getListener());
        }
        PropertyListener<List<FlowRule>> listener = new FlowRulePropertyListener(namespace);
        property.addListener(listener);
        PROPERTY_MAP.put(namespace, new NamespaceFlowProperty<>(namespace, property, listener));
        Set<Long> flowIdSet = NAMESPACE_FLOW_ID_MAP.get(namespace);
        if (flowIdSet == null) {
            resetNamespaceFlowIdMapFor(namespace);
        }
    }

    private static void registerPropertyInternal(/*@NonNull*/ String namespace, /*@Valid*/
                                                              Property<List<FlowRule>> property,
                                                              PropertyListener listener
    ) {
        NamespaceFlowProperty<FlowRule> oldProperty = PROPERTY_MAP.get(namespace);
        if (oldProperty != null) {
            oldProperty.getProperty().removeListener(oldProperty.getListener());
        }
        //PropertyListener<List<FlowRule>> listener = new FlowRulePropertyListener(namespace);
        property.addListener(listener);
        PROPERTY_MAP.put(namespace, new NamespaceFlowProperty<>(namespace, property, listener));
        Set<Long> flowIdSet = NAMESPACE_FLOW_ID_MAP.get(namespace);
        if (flowIdSet == null) {
            resetNamespaceFlowIdMapFor(namespace);
        }
    }

    public static void removeProperty(String namespace) {
        AssertUtil.notEmpty(namespace, "namespace cannot be empty");
        synchronized (UPDATE_LOCK) {
            NamespaceFlowProperty<FlowRule> property = PROPERTY_MAP.get(namespace);
            if (property != null) {
                property.getProperty().removeListener(property.getListener());
                PROPERTY_MAP.remove(namespace);
            }
//            RecordLog.info("[ClusterFlowRuleManager] Removing property from cluster flow rule manager"
//                    + " for namespace <{}>", namespace);
        }
    }

    private static void removePropertyListeners() {
        for (NamespaceFlowProperty<FlowRule> property : PROPERTY_MAP.values()) {
            property.getProperty().removeListener(property.getListener());
        }
    }

    private static void restorePropertyListeners() {
        for (NamespaceFlowProperty<FlowRule> p : PROPERTY_MAP.values()) {
            p.getProperty().removeListener(p.getListener());
            p.getProperty().addListener(p.getListener());
        }
    }

    public static FlowRule getFlowRuleById(Long id) {
        if (!ClusterRuleUtil.validId(id)) {
            return null;
        }
        return FLOW_RULES.get(id);
    }

    public static FlowRule getFlowRuleByResource(String resource) {

        return RESOURCE_RULES.get(resource);
    }

    public static Map<String, FlowRule> getResourceRules() {
        return RESOURCE_RULES;
    }

    public static Set<Long> getFlowIdSet(String namespace) {
        if (StringUtils.isEmpty(namespace)) {
            return new HashSet<>();
        }
        Set<Long> set = NAMESPACE_FLOW_ID_MAP.get(namespace);
        if (set == null) {
            return new HashSet<>();
        }
        return new HashSet<>(set);
    }

    public static List<FlowRule> getAllFlowRules() {
        return new ArrayList<>(FLOW_RULES.values());
    }

    public static List<FlowRule> getFlowRules(String namespace) {
        if (StringUtils.isEmpty(namespace)) {
            return new ArrayList<>();
        }
        List<FlowRule> rules = new ArrayList<>();
        Set<Long> flowIdSet = NAMESPACE_FLOW_ID_MAP.get(namespace);
        if (flowIdSet == null || flowIdSet.isEmpty()) {
            return rules;
        }
        for (Long flowId : flowIdSet) {
            FlowRule rule = FLOW_RULES.get(flowId);
            if (rule != null) {
                rules.add(rule);
            }
        }
        return rules;
    }

    /**
     * Load flow rules for a specific namespace. The former rules of the namespace will be replaced.
     *
     * @param namespace a valid namespace
     * @param rules     rule list
     */
    public static void loadRules(String namespace, List<FlowRule> rules) {
        AssertUtil.notEmpty(namespace, "namespace cannot be empty");
        NamespaceFlowProperty<FlowRule> property = PROPERTY_MAP.get(namespace);
        if (property != null) {
            property.getProperty().updateValue(rules);
        }
    }

    private static void resetNamespaceFlowIdMapFor(/*@Valid*/ String namespace) {
        NAMESPACE_FLOW_ID_MAP.put(namespace, new HashSet<Long>());
    }

//    private static void clearAndResetRulesConditional(/*@Valid*/ String namespace, Predicate<Long> predicate) {
//        Set<Long> oldIdSet = NAMESPACE_FLOW_ID_MAP.get(namespace);
//        if (oldIdSet != null && !oldIdSet.isEmpty()) {
//            for (Long flowId : oldIdSet) {
//                if (predicate.test(flowId)) {
//                    FLOW_RULES.remove(flowId);
//                    FLOW_NAMESPACE_MAP.remove(flowId);
//                    ClusterMetricStatistics.removeMetric(flowId);
//                    if (CurrentConcurrencyManager.containsFlowId(flowId)) {
//                        CurrentConcurrencyManager.remove(flowId);
//                    }
//                }
//            }
//            oldIdSet.clear();
//        }
//    }

    /**
     * Clear all rules of the provided namespace and reset map.
     *
     * @param namespace valid namespace
     */
    private static void clearAndResetRulesFor(/*@Valid*/ String namespace) {
        Set<Long> flowIdSet = NAMESPACE_FLOW_ID_MAP.get(namespace);
        if (flowIdSet != null && !flowIdSet.isEmpty()) {
            for (Long flowId : flowIdSet) {
                FLOW_RULES.remove(flowId);
                FLOW_NAMESPACE_MAP.remove(flowId);
                if (CurrentConcurrencyManager.containsFlowId(flowId)) {
                    CurrentConcurrencyManager.remove(flowId);
                }
            }
            flowIdSet.clear();
        } else {
            resetNamespaceFlowIdMapFor(namespace);
        }
    }

    /**
     * Get connected count for associated namespace of given {@code flowId}.
     *
     * @param flowId unique flow ID
     * @return connected count
     */
//    public static int getConnectedCount(long flowId) {
//        if (flowId <= 0) {
//            return 0;
//        }
//        String namespace = FLOW_NAMESPACE_MAP.get(flowId);
//        if (namespace == null) {
//            return 0;
//        }
//        return ConnectionManager.getConnectedCount(namespace);
//    }
    public static String getNamespace(long flowId) {
        return FLOW_NAMESPACE_MAP.get(flowId);
    }

    private static void applyClusterFlowRule(List<FlowRule> list, /*@Valid*/ String namespace) {
        if (list == null || list.isEmpty()) {
            clearAndResetRulesFor(namespace);
            return;
        }
        final ConcurrentHashMap<Long, FlowRule> ruleMap = new ConcurrentHashMap<>();

        Set<Long> flowIdSet = new HashSet<>();

        for (FlowRule rule : list) {
            if (!rule.isClusterMode()) {
                continue;
            }
            RESOURCE_RULES.put(rule.getResource(), rule);
//            if (!FlowRuleUtil.isValidRule(rule)) {
////                RecordLog.warn(
////                        "[ClusterFlowRuleManager] Ignoring invalid flow rule when loading new flow rules: " + rule);
//                continue;
//            }
//            if (StringUtils.isBlank(rule.getLimitApp())) {
//                rule.setLimitApp(RuleConstant.LIMIT_APP_DEFAULT);
//            }

            // Flow id should not be null after filtered.
            //    ClusterFlowConfig clusterConfig = rule.getClusterConfig();

//            Long flowId = clusterConfig.getFlowId();
//            if (flowId == null) {
//                continue;
//            }
//            ruleMap.put(flowId, rule);
//            FLOW_NAMESPACE_MAP.put(flowId, namespace);
//            flowIdSet.add(flowId);
//            if (!CurrentConcurrencyManager.containsFlowId(flowId)) {
//                CurrentConcurrencyManager.put(flowId, 0);
//            }

            // Prepare cluster metric from valid flow ID.
            ClusterMetricStatistics.putMetricIfAbsent(rule.getResource(),
                    new ClusterMetric(MetaInfo.PROPERTY_FLOW_CONTROL_SAMPLE_COUNT, MetaInfo.PROPERTY_FLOW_CONTROL_SAMPLE_INTERVAL));
        }

        // Cleanup unused cluster metrics.
//        clearAndResetRulesConditional(namespace, new Predicate<Long>() {
//            @Override
//            public boolean test(Long flowId) {
//                return !ruleMap.containsKey(flowId);
//            }
//        });

        FLOW_RULES.putAll(ruleMap);
        NAMESPACE_FLOW_ID_MAP.put(namespace, flowIdSet);

    }

    private static final class FlowRulePropertyListener implements PropertyListener<List<FlowRule>> {

        private final String namespace;

        public FlowRulePropertyListener(String namespace) {
            this.namespace = namespace;
        }

        @Override
        public synchronized void configUpdate(List<FlowRule> conf) {
            logger.info("config update {}", conf);
            applyClusterFlowRule(conf, namespace);
//            RecordLog.info("[ClusterFlowRuleManager] Cluster flow rules received for namespace <{}>: {}",
//                    namespace, FLOW_RULES);
        }

        @Override
        public synchronized void configLoad(List<FlowRule> conf) {

            applyClusterFlowRule(conf, namespace);
            logger.info("flow rule load {}", JsonUtil.formatJson(JsonUtil.object2Json(RESOURCE_RULES)));
//            RecordLog.info("[ClusterFlowRuleManager] Cluster flow rules loaded for namespace <{}>: {}",
//                    namespace, FLOW_RULES);
        }
    }
}
