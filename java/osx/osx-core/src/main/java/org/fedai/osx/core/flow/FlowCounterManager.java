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
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.fedai.osx.core.utils.GetSystemInfo;
import org.fedai.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;


public class FlowCounterManager {

    public static final boolean USE_PID = true;
    private static final String DEFAULT_CONFIG_FILE = "/test" + File.separator + ".fate" + File.separator + "flowRules.json";
    Logger logger = LoggerFactory.getLogger(FlowCounterManager.class);
    String appName;
    MetricSearcher metricSearcher;
    File file;
    MetricReport metricReport;
    ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);
    Map<String, Double> sourceQpsAllowMap = new HashMap<>();
    private ConcurrentHashMap<String, FlowCounter> passMap = new ConcurrentHashMap<>();
    public FlowCounterManager() {
        this("default");
    }

    public FlowCounterManager(String appName) {
        this(appName, false);
    }

    public FlowCounterManager(String appName, Boolean countModelRequest) {
        this.appName = appName;
        String baseFileName = appName + "-metrics.log";
        if (USE_PID) {
            baseFileName += ".pid" + GetSystemInfo.getPid();

        }
        metricSearcher = new MetricSearcher(MetricWriter.METRIC_BASE_DIR, baseFileName);
        metricReport = new FileMetricReport(appName);
//        if (countModelRequest) {
//            modelMetricReport = new FileMetricReport("model");
//         //   modelMetricSearcher = new MetricSearcher(MetricWriter.METRIC_BASE_DIR, modelFileName);
//        }
    }

//    public static void main(String[] args) throws IOException {
//        MetaInfo.PROPERTY_ROOT_PATH = new File("").getCanonicalPath();
//
//        FlowCounterManager flowCounterManager = new FlowCounterManager("test");
//        flowCounterManager.setMetricReport(new FileMetricReport("Test"));
//        flowCounterManager.setMetricSearcher(new MetricSearcher(MetricWriter.METRIC_BASE_DIR, "Test" + "-metrics.log.pid" + GetSystemInfo.getPid()));
//        flowCounterManager.startReport();
//        flowCounterManager.file = new File(MetaInfo.PROPERTY_ROOT_PATH + File.separator + ".fate" + File.separator + "flowRules.json");
//        flowCounterManager.initialize();
//
//        int i = 0;
//        while (true) {
//            flowCounterManager.setAllowQps("source-" + i, i);
//            i++;
//            try {
//                Thread.sleep(5000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
//        /*while (true) {
//            flowCounterManager.pass("M_test");
//            try {
//                Thread.sleep(100);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }*/
//    }

    public MetricSearcher getMetricSearcher() {
        return metricSearcher;
    }

    public void setMetricSearcher(MetricSearcher metricSearcher) {
        this.metricSearcher = metricSearcher;
    }

    public List<MetricNode> queryMetrics(long beginTimeMs, long endTimeMs, String identity) {
        try {
            return metricSearcher.findByTimeAndResource(beginTimeMs, endTimeMs, identity);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public List<MetricNode> queryAllMetrics(long beginTimeMs, int size) {
        try {
            return metricSearcher.find(beginTimeMs, size);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

//    public List<MetricNode> queryModelMetrics(long beginTimeMs, long endTimeMs, String identity) {
//
//        try {
//            return modelMetricSearcher.findByTimeAndResource(beginTimeMs, endTimeMs, identity);
//        } catch (Exception e) {
//            logger.error("find model metric error", e);
//            throw new RuntimeException("find model metric error");
//        }
//    }

//    public List<MetricNode> queryAllModelMetrics(long beginTimeMs, int size) {
//        try {
//            return modelMetricSearcher.find(beginTimeMs, size);
//        } catch (Exception e) {
//            logger.error("find mode metric error", e);
//            throw new RuntimeException("find mode metric error");
//        }
//    }

    public MetricReport getMetricReport() {
        return metricReport;
    }

    public void setMetricReport(MetricReport metricReport) {
        this.metricReport = metricReport;
    }

    public boolean pass(String sourceName, int times) {
        FlowCounter flowCounter = passMap.get(sourceName);
        if (flowCounter == null) {
            Double allowedQps = getAllowedQps(sourceName);
            flowCounter = passMap.putIfAbsent(sourceName, new FlowCounter(allowedQps != null ? allowedQps : Integer.MAX_VALUE));
            if (flowCounter == null) {
                flowCounter = passMap.get(sourceName);
            }
        }
//        logger.info("source {} pass {}",sourceName,times);
        return flowCounter.tryPass(times);
    }

//    public boolean success(String sourceName, int times) {
//        FlowCounter flowCounter = successMap.get(sourceName);
//        if (flowCounter == null) {
//            flowCounter = successMap.putIfAbsent(sourceName, new FlowCounter(Integer.MAX_VALUE));
//            if (flowCounter == null) {
//                flowCounter = successMap.get(sourceName);
//            }
//        }
//        return flowCounter.tryPass(times);
//    }

//    public boolean block(String sourceName, int times) {
//        FlowCounter flowCounter = blockMap.get(sourceName);
//        if (flowCounter == null) {
//            flowCounter = blockMap.putIfAbsent(sourceName, new FlowCounter(Integer.MAX_VALUE));
//            if (flowCounter == null) {
//                flowCounter = blockMap.get(sourceName);
//            }
//        }
//        return flowCounter.tryPass(times);
//    }
//
//    public boolean exception(String sourceName, int times) {
//        FlowCounter flowCounter = exceptionMap.get(sourceName);
//        if (flowCounter == null) {
//            flowCounter = exceptionMap.putIfAbsent(sourceName, new FlowCounter(Integer.MAX_VALUE));
//            if (flowCounter == null) {
//                flowCounter = exceptionMap.get(sourceName);
//            }
//        }
//        return flowCounter.tryPass(times);
//    }

    public void startReport() {
//        init();
        executor.scheduleAtFixedRate(() -> {
            //logger.info("startReport");
            long current = TimeUtil.currentTimeMillis();
            List<MetricNode> reportList = Lists.newArrayList();
            List<MetricNode> modelReportList = Lists.newArrayList();
            passMap.forEach((sourceName, flowCounter) -> {
//                FlowCounter successCounter = successMap.get(sourceName);
//                FlowCounter blockCounter = blockMap.get(sourceName);
//                FlowCounter exceptionCounter = exceptionMap.get(sourceName);
                MetricNode metricNode = new MetricNode();
                metricNode.setTimestamp(current);
                metricNode.setResource(sourceName);
                metricNode.setPassQps(flowCounter.getSum());
//                metricNode.setBlockQps(blockCounter != null ? new Double(blockCounter.getQps()).longValue() : 0);
//                metricNode.setExceptionQps(exceptionCounter != null ? new Double(exceptionCounter.getQps()).longValue() : 0);
//                metricNode.setSuccessQps(successCounter != null ? new Double(successCounter.getQps()).longValue() : 0);

                reportList.add(metricNode);


                //modelReportList.add(metricNode);

            });
            //logger.info("try to report {}",reportList);
            metricReport.report(reportList);
//            if (modelMetricReport != null) {
//                modelMetricReport.report(modelReportList);
//            }
        }, 0, 1, TimeUnit.SECONDS);
    }

    public void rmAllFiles() {
        try {
//            if (modelMetricReport instanceof FileMetricReport) {
//                FileMetricReport fileMetricReport = (FileMetricReport) modelMetricReport;
//                fileMetricReport.rmAllFile();
//            }
            if (metricReport instanceof FileMetricReport) {
                FileMetricReport fileMetricReport = (FileMetricReport) metricReport;
                fileMetricReport.rmAllFile();
            }
        } catch (Exception e) {
            logger.error("remove metric file error");
        }
    }

    /**
     * init rules
     */
    private void initialize() throws IOException {
        file = new File(DEFAULT_CONFIG_FILE);
        logger.info("try to load flow counter rules, {}", file.getAbsolutePath());

        if (file.exists()) {
            String result = "";
            try (
                    FileInputStream fileInputStream = new FileInputStream(file);
                    InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
                    BufferedReader reader = new BufferedReader(inputStreamReader)
            ) {
                String tempString;
                while ((tempString = reader.readLine()) != null) {
                    result += tempString;
                }

                List<Map> list = JsonUtil.json2Object(result, new TypeReference<List<Map>>() {
                });
                if (list != null) {
                    list.forEach(map -> {
                        sourceQpsAllowMap.put((String) map.get("source"), Double.valueOf(String.valueOf(map.get("allow_qps"))));
                    });
                }
            } catch (IOException e) {
                logger.error("load flow counter rules failed, use default setting, cause by: {}", e.getMessage());
            }
            logger.info("load flow counter rules success");
        }
    }

    private void store(File file, byte[] data) {
        try {
            if (!file.exists() && file.getParentFile() != null && !file.getParentFile().exists()) {
                if (!file.getParentFile().mkdirs()) {
                    throw new IllegalArgumentException("invalid flow control cache file " + file + ", cause: Failed to create directory " + file.getParentFile() + "!");
                }
            }
            if (!file.exists()) {
                file.createNewFile();
            }
            try (FileOutputStream outputFile = new FileOutputStream(file)) {
                outputFile.write(data);
            }
        } catch (Throwable e) {
            logger.error("store rules file failed, cause: {}", e.getMessage());
        }
    }

    public Double getAllowedQps(String sourceName) {
        return sourceQpsAllowMap.get(sourceName);
    }

    public void setAllowQps(String sourceName, double allowQps) {
        logger.info("update {} allowed qps to {}", sourceName, allowQps);
        sourceQpsAllowMap.put(sourceName, allowQps);

        List<Map> list = sourceQpsAllowMap.entrySet().stream()
                .sorted(Comparator.comparing(Map.Entry::getKey))
                .map(entry -> {
                    Map map = Maps.newHashMap();
                    map.put("source", entry.getKey());
                    map.put("allow_qps", entry.getValue());
                    return map;
                })
                .collect(Collectors.toList());

        this.store(file, JsonUtil.object2Json(list).getBytes());

        // 更新FlowCounter
        FlowCounter flowCounter = passMap.get(sourceName);
        if (flowCounter != null) {
            flowCounter.setQpsAllowed(allowQps);
        }
    }

    public void destroy() {
        System.err.println("try to destroy flow counter");
        if (this != null) {
            this.rmAllFiles();
        }
    }

//    @Override
//    public void onApplicationEvent(ApplicationReadyEvent applicationReadyEvent) {
//        startReport();
//    }
}
