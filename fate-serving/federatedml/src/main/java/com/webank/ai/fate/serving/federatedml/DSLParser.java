package com.webank.ai.fate.serving.federatedml;

import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.PipelineProto;
import com.webank.ai.fate.serving.federatedml.model.BaseModel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONObject;
import org.json.JSONArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;

public class DSLParser {
    private HashMap<String, String> componentModuleMap = new HashMap<String, String>();
    private HashMap<String, Integer> componentIds = new HashMap<String, Integer>();
    private HashMap<String, List<String>> downStream = new HashMap<String, List<String>>();
    private HashMap<Integer, HashSet<Integer>> upInputs = new HashMap<Integer, HashSet<Integer>>();
    private ArrayList<String> topoRankComponent = new ArrayList<String>();
    private String modelPackage = "com.webank.ai.fate.serving.federatedml.model";
    private static final Logger LOGGER = LogManager.getLogger();
    
    
    public int parseDagFromDSL(String jsonStr) {
        LOGGER.info("start parse dag from dsl");
        try {
            JSONObject dsl = new JSONObject(jsonStr);
            JSONObject components = dsl.getJSONObject("components");
            
            LOGGER.info("start topo sort");
            topoSort(components, this.topoRankComponent);
             
            LOGGER.info("components size is {}", this.topoRankComponent.size());
            for (int i = 0; i < this.topoRankComponent.size(); ++i) {
                this.componentIds.put(this.topoRankComponent.get(i), i);
            }
            
            for (int i = 0; i < topoRankComponent.size(); ++i) {
                String componentName = topoRankComponent.get(i);
                LOGGER.info("component is {}", componentName);
                JSONObject component = components.getJSONObject(componentName);
                String[] codePath = ((String)component.get("CodePath")).split("/", -1);
                LOGGER.info("code path splits is {}", codePath);
                String module = codePath[codePath.length - 1];
                LOGGER.info("module is {}", module);
                componentModuleMap.put(componentName, module);
        
                JSONObject upData = component.getJSONObject("input").getJSONObject("data");
                if (upData != null) {
                    int componentId = this.componentIds.get(componentName);
                    Iterator<String> dataKeyIterator = upData.keys();
                    while (dataKeyIterator.hasNext()) {
                        String dataKey = dataKeyIterator.next();
                        JSONArray data = upData.getJSONArray(dataKey);
                        for (int j = 0; j < data.length(); j++) {
                            String upComponent = data.getString(j).split("\\.", -1)[0];
                            if (!upInputs.containsKey(componentId)) {
                                upInputs.put(componentId, new HashSet<Integer>());
                            }

                            if (upComponent.equals("args")) {
                                upInputs.get(componentId).add(-1);
                            } else {
                                upInputs.get(componentId).add(this.componentIds.get(upComponent));
                            }
                        }

                    }
                }
            }
        
        } catch (Exception ex) {
            ex.printStackTrace();
            LOGGER.info("DSLParser init catch error:{}", ex);
        }
        LOGGER.info("Finish init DSLParser");
        return StatusCode.OK;
    }
    
    public void topoSort(JSONObject components, ArrayList<String> topoRankComponent) {
        Stack<Integer> stk = new Stack();
        HashMap<String, Integer> componentIndexMapping = new HashMap<String, Integer>();
        ArrayList<String> componentList = new ArrayList<String>();
        int index = 0;
        Iterator<String> componentNames = components.keys();
        
        while (componentNames.hasNext()) {
            String componentName = componentNames.next();
            componentIndexMapping.put(componentName, index);
            ++index;
            componentList.add(componentName);
        }
        
        int inDegree[] = new int[index];
        HashMap<Integer, ArrayList<Integer> > edges = new HashMap<Integer, ArrayList<Integer> >();
        
        for (int i = 0; i < componentList.size(); ++i) {
            String componentName = componentList.get(i);
            JSONObject component = components.getJSONObject(componentName);
            JSONObject upData = component.getJSONObject("input").getJSONObject("data");
            Integer componentId = componentIndexMapping.get(componentName);
            if (upData != null) {
                Iterator<String> dataKeyIterator = upData.keys();
                while (dataKeyIterator.hasNext()) {
                    String dataKey = dataKeyIterator.next();
                    JSONArray data = upData.getJSONArray(dataKey);
                    for (int j = 0; j < data.length(); j++) {
                        String upComponent = data.getString(j).split("\\.", -1)[0];
                        int upComponentId = -1;
                        if (!upComponent.equals("args")) {
                            upComponentId = componentIndexMapping.get(upComponent);
                        }
    
                        if (upComponentId != -1) {			
                            if (!edges.containsKey(upComponentId)) {
                                edges.put(upComponentId, new ArrayList<Integer>());
                            }
                            inDegree[componentId]++;
                            edges.get(upComponentId).add(componentId);
                        }
                    }
                }
            }
        }
        
        LOGGER.info("end of construct edges"); 
        for (int i = 0; i < index; i++) {
            if (inDegree[i] == 0) {
                stk.push(i);
            }
        }
        
        while (!stk.empty()) {
            Integer vertex = stk.pop();
            topoRankComponent.add(componentList.get(vertex));
            ArrayList<Integer> adjV = edges.get(vertex);
            if (adjV == null) {
                continue;
            }
            for (int i = 0; i < adjV.size(); ++i) {
                Integer downV = adjV.get(i);
                --inDegree[downV];
                if (inDegree[downV] == 0) {
                    stk.push(downV);
                }
            }
        }
        
        LOGGER.info("end of topo"); 
    }

    public HashMap<String, String> getComponentModuleMap() {
        return this.componentModuleMap;
    }
    
    public ArrayList<String> getAllComponent() {
        return this.topoRankComponent;
    }
    
    public HashSet<Integer> getUpInputComponents(int idx) {
        if (this.upInputs.containsKey(idx)) {
            return this.upInputs.get(idx);
        } else {
            return null;
        }
    }
   
}
