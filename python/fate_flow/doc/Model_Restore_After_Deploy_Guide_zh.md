# 模型备份恢复指南

## 备份模型文件目录

1. 1.4.x版本: **$FATE_PATH/python/model_local_cache**
2. 1.5.x版本: **$FATE_PATH/model_local_cache**

## 还原模型文件目录
1. 1.4.x版本，**model_local_cache**拷贝到**$FATE_PATH/python/**
2. 1.5.x版本，**model_local_cache**拷贝到**$FATE_PATH/**

### 模型迁移(若部署完成后，原有party id发生变化，则需要操作)
参考[模型迁移指南](Fate_Flow_Model_Migration_Guide_zh.md)
