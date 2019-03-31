from arch.api.model_manager import core
# from arch.api.proto.imputer_pb2 import imputer
# from arch.api.proto.outlier_pb2 import Outlier
# from arch.api.proto.scale_pb2 import Scale
# from arch.api.proto.data_transform_pb2 import ScaleObject
# from arch.api.proto.data_transform_server_pb2 import DataTransformServer
from arch.api.utils import log_utils
from federatedml.feature.min_max_scaler import MinMaxScaler
from federatedml.feature.standard_scaler import StandardScaler
# from federatedml.param import DataTransformParam
from federatedml.util.param_extract import ParamExtract
from federatedml.util import consts

LOGGER = log_utils.getLogger()

class Scaler(object):
    def __init__(self, scale_param):
        self.scale_param = scale_param

    def fit(self, data):
        LOGGER.info("Start scale data fit ...")
        scale_value_results = []

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler(mode=self.scale_param.mode, area=self.scale_param.area,
                                        feat_upper=self.scale_param.feat_upper, feat_lower=self.scale_param.feat_lower, 
                                        out_upper=self.scale_param.out_upper, out_lower=self.scale_param.out_lower)
                
            data, cols_scale_value = min_max_scaler.fit(data)
            scale_value_results.append(cols_scale_value)

        elif self.data_transform_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler(with_mean=self.scale_param.with_mean, with_std=self.scale_param.with_std)
            data, mean, std = standard_scaler.fit(data)                              
            scale_value_results.append(mean)
            scale_value_results.append(std)

        else:
            LOGGER.info("Scale method is {}, do nothing and return!".format(self.scale_param.method))

        LOGGER.info("End fit data ...")
        return data, scale_value_results
    
    def transform(self, data, fit_config):
        LOGGER.info("Start scale data transform ...")
        
        if len(fit_config) == 0:
            LOGGER.warning("length fit_config is 0, can not do transform, do nothing and return")

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler()
            data = min_max_scaler.transform(data, fit_config[0])
        elif self.data_scale_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler()
            data = standard_scaler.transform(data, mean=fit_config[0], std=fit_config[1])
        else:
            LOGGER.info("DataTransform method is {}, do nothing and return!".format(self.data_transform_param.method))
            
        LOGGER.info("End transform data ...")
        
        return data

    # @staticmethod
    # def data_transform_proto_convert_to_serving(data_transform_proto, index_to_key):
    #     data_transform_server_proto = DataTransformServer()
    #     
    #     data_transform_server_proto.missing_fill = data_transform_proto.missing_fill
    #     if data_transform_proto.missing_fill:
    #         for value in data_transform_proto.missing_value:
    #             data_transform_server_proto.missing_value.append(value)

    #         data_transform_server_proto.missing_replace_method = data_transform_proto.missing_replace_method
    #             
    #         for k in data_transform_proto.missing_replace_value:
    #             data_transform_server_proto.missing_replace_value[index_to_key[k]] = data_transform_proto.missing_replace_value[k]

    #     data_transform_server_proto.outlier_replace = data_transform_proto.outlier_replace
    #     if data_transform_proto.outlier_replace:

    #         for value in data_transform_proto.outlier_value:
    #             data_transform_server_proto.outlier_value.append(value)

    #         data_transform_server_proto.outlier_replace_method = data_transform_proto.outlier_replace_method
    #         
    #         for k in data_transform_proto.outlier_replace_value:
    #             data_transform_server_proto.outlier_replace_value[index_to_key[k]] = data_transform_proto.outlier_replace_value[k]
    #     
    #     data_transform_server_proto.is_scale = data_transform_proto.is_scale
    #     if data_transform_proto.is_scale:
    #         data_transform_server_proto.scale_method = data_transform_proto.scale_method
    #         
    #         for k in data_transform_proto.scale_replace_value:
    #             scale = data_transform_server_proto.scale_replace_value[index_to_key[k]]
    #             scale.feat_lower = data_transform_proto.scale_replace_value[k].feat_lower
    #             scale.feat_upper = data_transform_proto.scale_replace_value[k].feat_upper
    #             scale.out_lower = data_transform_proto.scale_replace_value[k].out_lower
    #             scale.out_upper = data_transform_proto.scale_replace_value[k].out_upper
    #             
    #             scale.mean = data_transform_proto.scale_replace_value[k].mean
    #             scale.std_var = data_transform_proto.scale_replace_value[k].std_var
    #     
    #     return data_transform_server_proto
