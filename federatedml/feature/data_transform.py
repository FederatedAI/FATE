from arch.api.model_manager import core
from arch.api.proto.data_transform_pb2 import DataTransform as DataTransformProto
from arch.api.proto.data_transform_server_pb2 import DataTransformServer
from arch.api.utils import log_utils
from federatedml.feature.min_max_scale import MinMaxScale
from federatedml.feature.standard_scale import StandardScale
from federatedml.param.param import DataTransformParam
from federatedml.util import consts
from federatedml.util.param_extract import ParamExtract

LOGGER = log_utils.getLogger()


class DataTransform(object):
    def __init__(self, config):
        data_transform_params = DataTransformParam()
        self.data_transform_params = ParamExtract.parse_param_from_config(data_transform_params, config)

    def fit_transform(self, data, fit_config=None):
        LOGGER.info("Start transform data ...")

        if fit_config is None:
            LOGGER.info("fit data ...")
            data_transform_proto = DataTransformProto()
            cols_transform_value = None

            # try:
            core.read_model("data_transform", data_transform_proto)

            # except:
            #    LOGGER.debug("Could not read data_transform from model_manager, set a new one")

            if self.data_transform_params.method == consts.MINMAXSCALE:
                min_max_scaler = MinMaxScale(mode=self.data_transform_params.mode, area=self.data_transform_params.area,
                                             feat_upper=self.data_transform_params.feat_upper,
                                             feat_lower=self.data_transform_params.feat_lower,
                                             out_upper=self.data_transform_params.out_upper,
                                             out_lower=self.data_transform_params.out_lower)

                data, cols_transform_value = min_max_scaler.fit(data)
                data_transform_proto.is_scale = True
                data_transform_proto.scale_method = consts.MINMAXSCALE

                for i in range(len(cols_transform_value)):
                    scale_obj = data_transform_proto.scale_replace_value[i]
                    scale_obj.feat_lower = cols_transform_value[i][0]
                    scale_obj.feat_upper = cols_transform_value[i][1]
                    scale_obj.out_lower = cols_transform_value[i][2]
                    scale_obj.out_upper = cols_transform_value[i][3]

            elif self.data_transform_params.method == consts.STANDARDSCALE:
                standard_scaler = StandardScale(with_mean=self.data_transform_params.with_mean,
                                                with_std=self.data_transform_params.with_std)
                data, mean, std = standard_scaler.fit(data)
                data_transform_proto.is_scale = True
                data_transform_proto.scale_method = consts.STANDARDSCALE

                for i in range(len(mean)):
                    scale_obj = data_transform_proto.scale_replace_value[i]
                    scale_obj.mean = mean[i]
                    scale_obj.std_var = std[i]

            else:
                LOGGER.info(
                    "DataTransform method is {}, do nothing and return!".format(self.data_transform_params.method))
                data_transform_proto.is_scale = False

            core.save_model("data_transform", data_transform_proto)

        else:
            if self.data_transform_params.method == consts.MINMAXSCALE:
                min_max_scaler = MinMaxScale()
                data = min_max_scaler.transform(data, fit_config)
            elif self.data_transform_params.method == consts.STANDARDSCALE:
                standard_scaler = StandardScale()
                data = standard_scaler.transform(data, fit_config)
            else:
                LOGGER.info(
                    "DataTransform method is {}, do nothing and return!".format(self.data_transform_params.method))

            cols_transform_value = fit_config

        LOGGER.info("End transform data ...")

        return data, cols_transform_value

    @staticmethod
    def data_transform_proto_convert_to_serving(data_transform_proto, index_to_key):
        data_transform_server_proto = DataTransformServer()

        data_transform_server_proto.missing_fill = data_transform_proto.missing_fill
        if data_transform_proto.missing_fill:
            for value in data_transform_proto.missing_value:
                data_transform_server_proto.missing_value.append(value)

            data_transform_server_proto.missing_replace_method = data_transform_proto.missing_replace_method

            for k in data_transform_proto.missing_replace_value:
                data_transform_server_proto.missing_replace_value[index_to_key[k]] = \
                data_transform_proto.missing_replace_value[k]

        data_transform_server_proto.outlier_replace = data_transform_proto.outlier_replace
        if data_transform_proto.outlier_replace:

            for value in data_transform_proto.outlier_value:
                data_transform_server_proto.outlier_value.append(value)

            data_transform_server_proto.outlier_replace_method = data_transform_proto.outlier_replace_method

            for k in data_transform_proto.outlier_replace_value:
                data_transform_server_proto.outlier_replace_value[index_to_key[k]] = \
                data_transform_proto.outlier_replace_value[k]

        data_transform_server_proto.is_scale = data_transform_proto.is_scale
        if data_transform_proto.is_scale:
            data_transform_server_proto.scale_method = data_transform_proto.scale_method

            for k in data_transform_proto.scale_replace_value:
                scale = data_transform_server_proto.scale_replace_value[index_to_key[k]]
                scale.feat_lower = data_transform_proto.scale_replace_value[k].feat_lower
                scale.feat_upper = data_transform_proto.scale_replace_value[k].feat_upper
                scale.out_lower = data_transform_proto.scale_replace_value[k].out_lower
                scale.out_upper = data_transform_proto.scale_replace_value[k].out_upper

                scale.mean = data_transform_proto.scale_replace_value[k].mean
                scale.std_var = data_transform_proto.scale_replace_value[k].std_var

        return data_transform_server_proto
