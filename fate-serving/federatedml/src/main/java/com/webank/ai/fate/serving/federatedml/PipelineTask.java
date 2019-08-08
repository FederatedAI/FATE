package com.webank.ai.fate.serving.federatedml;

import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.constant.StatusCode;
import com.webank.ai.fate.core.mlmodel.buffer.PipelineProto;
import com.webank.ai.fate.serving.core.bean.Context;
import com.webank.ai.fate.serving.federatedml.model.BaseModel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PipelineTask {
    private List<BaseModel> pipeLineNode = new ArrayList<>();
    private String modelPackage = "com.webank.ai.fate.serving.federatedml.model";
    private static final Logger LOGGER = LogManager.getLogger();


    public int initModel(Map<String, byte[]> modelProtoMap) {
        LOGGER.info("start init Pipeline");
        try {
            String pipelineProtoName = "Pipeline";
            PipelineProto.Pipeline pipeLineProto = PipelineProto.Pipeline.parseFrom(modelProtoMap.get(pipelineProtoName));
            LOGGER.info("Finish get Pipeline proto");
            List<String> pipeLineMeta = pipeLineProto.getNodeMetaList();
            List<String> pipeLineParam = pipeLineProto.getNodeParamList();

            for (int i = 0; i < pipeLineMeta.size(); i++) {
                String className = pipeLineMeta.get(i).split("\\.")[0];
                LOGGER.info("Start get className:{}", className);
                try {
                    Class modelClass = Class.forName(this.modelPackage + "." + className);
                    BaseModel mlNode = (BaseModel) modelClass.getConstructor().newInstance();
                    byte[] protoMeta = modelProtoMap.get(pipeLineMeta.get(i));
                    byte[] protoParam = modelProtoMap.get(pipeLineParam.get(i));
                    mlNode.initModel(protoMeta, protoParam);

                    pipeLineNode.add(mlNode);
                    LOGGER.info(" Add class {} to pipeline task list", className);
                } catch (Exception ex) {
                    LOGGER.warn("Can not instance {} class", className);
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            LOGGER.info("Pipeline init catch error:{}", ex);
        }
        LOGGER.info("Finish init Pipeline");
        return StatusCode.OK;
    }

    public Map<String, Object> predict(Context context , Map<String, Object> inputData, Map<String, Object> predictParams) {
        LOGGER.info("Start Pipeline predict use {} model node.", this.pipeLineNode.size());
        for (int i = 0; i < this.pipeLineNode.size(); i++) {
            LOGGER.info(this.pipeLineNode.get(i).getClass().getName());
            inputData = this.pipeLineNode.get(i).handlePredict(context,inputData, predictParams);
            LOGGER.info("finish mlNone:{}", i);
        }
        ReturnResult federatedResult = context.getFederatedResult();
        if(federatedResult!=null) {
            inputData.put("retcode", federatedResult.getRetcode());
        }

        LOGGER.info("Finish Pipeline predict");
        return inputData;
    }
}
