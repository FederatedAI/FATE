from arch.api.utils import log_utils
from arch.api.proto import feature_scale_meta_pb2
from arch.api.proto import feature_scale_param_pb2
from arch.api.model_manager import manager as model_manager
from federatedml.feature.min_max_scaler import MinMaxScaler
from federatedml.feature.standard_scaler import StandardScaler
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Scaler(object):
    """
    The Scaler class is used to data scale. MinMaxScale and StandardScale is supported now
    """

    def __init__(self, scale_param):
        """
        Parameters
        ----------
        scale_param: object, the parameters of scale
        """
        self.scale_param = scale_param
        self.mean = None
        self.std = None
        self.std_scale_column_idx = None
        self.class_name = self.__class__.__name__
        self.cols_scale_res = None

    def fit(self, data):
        """
        Apply scale for input data
        Parameters
        ----------
        data: data_instance, input data

        Returns
        ----------
        data:data_instance, data after scale
        scale_value_results: list, the fit results information of scale
        """
        LOGGER.info("Start scale data fit ...")

        self.header = data.schema.get('header')

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler(mode=self.scale_param.mode, area=self.scale_param.area,
                                          scale_column_idx=self.scale_param.scale_column_idx,
                                          feat_upper=self.scale_param.feat_upper,
                                          feat_lower=self.scale_param.feat_lower,
                                          out_upper=self.scale_param.out_upper, out_lower=self.scale_param.out_lower)

            data, cols_scale_value = min_max_scaler.fit(data)
            self.cols_scale_res = cols_scale_value

        elif self.scale_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler(area=self.scale_param.area,
                                             scale_column_idx=self.scale_param.scale_column_idx,
                                             with_mean=self.scale_param.with_mean, with_std=self.scale_param.with_std)
            data, cols_scale_value = standard_scaler.fit(data)
            self.mean = cols_scale_value[0]
            self.std = cols_scale_value[1]
            self.std_scale_column_idx = cols_scale_value[2]
            self.cols_scale_res = cols_scale_value
        else:
            LOGGER.info("Scale method is {}, do nothing and return!".format(self.scale_param.method))

        data.schema['header'] = self.header
        LOGGER.info("End fit data ...")
        return data, self.cols_scale_res

    def transform(self, data, fit_config):
        """
        Transform input data using scale with fit results
        Parameters
        ----------
        data: data_instance, input data
        fit_config: list, the fit results information of scale

        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        LOGGER.info("Start scale data transform ...")
        self.header = data.schema.get('header')
        if len(fit_config) == 0:
            LOGGER.warning("length fit_config is 0, can not do transform, do nothing and return")

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler()
            data = min_max_scaler.transform(data, fit_config)
        elif self.scale_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler()
            data = standard_scaler.transform(data, mean=fit_config[0], scale=fit_config[1],
                                             scale_column_idx=fit_config[2])
        else:
            LOGGER.info("DataTransform method is {}, do nothing and return!".format(self.scale_param.method))

        data.schema['header'] = self.header
        LOGGER.info("End transform data ...")

        return data

    def _save_min_max_meta(self, name, namespace):
        if self.scale_param.area == consts.ALL:
            LOGGER.debug("save_min_max_meta with area is all")
            feat_upper = "None" if self.scale_param.feat_upper is None else str(self.scale_param.feat_upper)
            feat_lower = "None" if self.scale_param.feat_lower is None else str(self.scale_param.feat_lower)
            out_upper = "None" if self.scale_param.out_upper is None else str(self.scale_param.out_upper)
            out_lower = "None" if self.scale_param.out_lower is None else str(self.scale_param.out_lower)

            min_max_scale_meta = feature_scale_meta_pb2.MinMaxScaleMeta(feat_upper=feat_upper,
                                                                        feat_lower=feat_lower,
                                                                        out_upper=out_upper,
                                                                        out_lower=out_lower)

            minmax_scale_meta = {"0": min_max_scale_meta}
            meta_protobuf_obj = feature_scale_meta_pb2.ScaleMeta(is_scale=True,
                                                                 strategy=self.scale_param.method,
                                                                 minmax_scale_meta=minmax_scale_meta)
        else:
            LOGGER.debug("save_min_max_meta with area is {}".format(self.scale_param.mode))
            meta_protobuf_obj = feature_scale_meta_pb2.ScaleMeta(is_scale=True)

        buffer_type = "{}.meta".format(self.class_name)
        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_min_max_model(self, name, namespace):
        """
        Save MinMaxScaler param and meta
        Parameters
        ----------
        name: str, save DTable table_name
        namespace: str, DTable namespace

        Returns
        ----------
        tuple,
            include meta string name and param string name
        """
        meta_buffer_type = self._save_min_max_meta(name, namespace)

        cols_scale_value = self.cols_scale_res[0]
        scale_column_idx = self.cols_scale_res[1]

        min_max_scale_param_dict = {}
        if cols_scale_value is not None:
            for i in range(len(self.header)):
                if i in scale_column_idx:
                    feat_lower = cols_scale_value[i][0]
                    feat_upper = cols_scale_value[i][1]
                    out_lower = cols_scale_value[i][2]
                    out_upper = cols_scale_value[i][3]
                    param_obj = feature_scale_param_pb2.MinMaxScaleParam(feat_upper=feat_upper,
                                                                         feat_lower=feat_lower,
                                                                         out_upper=out_upper,
                                                                         out_lower=out_lower)
                    min_max_scale_param_dict[self.header[i]] = param_obj

        param_protobuf_obj = feature_scale_param_pb2.ScaleParam(minmax_scale_param=min_max_scale_param_dict)
        param_buffer_type = "{}.param".format(self.class_name)

        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=param_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return [(meta_buffer_type, param_buffer_type)]

    def _save_standard_scale_meta(self, name, namespace):
        with_mean = self.scale_param.with_mean
        with_std = self.scale_param.with_std

        standard_scale_meta = feature_scale_meta_pb2.StandardScaleMeta(with_mean=with_mean, with_std=with_std)

        meta_protobuf_obj = feature_scale_meta_pb2.ScaleMeta(is_scale=True,
                                                             strategy=self.scale_param.method,
                                                             standard_scale_meta=standard_scale_meta)

        buffer_type = "{}.meta".format(self.class_name)
        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=meta_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return buffer_type

    def save_standard_scale_model(self, name, namespace):
        """
        Save StandardScaler param and meta
        Parameters
        ----------
        name: str, save DTable table_name
        namespace: str, DTable namespace

        Returns
        ----------
        tuple,
            include meta string name and param string name
        """
        meta_buffer_type = self._save_standard_scale_meta(name, namespace)

        standard_scale_param_dict = {}
        for i in range(len(self.header)):
            if i in self.std_scale_column_idx:
                mean = self.mean[i]
                std = self.std[i]

                param_obj = feature_scale_param_pb2.StandardScaleParam(mean=mean, scale=std)
                standard_scale_param_dict[self.header[i]] = param_obj

        param_protobuf_obj = feature_scale_param_pb2.ScaleParam(standard_scale_param=standard_scale_param_dict)
        param_buffer_type = "{}.param".format(self.class_name)

        model_manager.save_model(buffer_type=param_buffer_type,
                                 proto_buffer=param_protobuf_obj,
                                 name=name,
                                 namespace=namespace)
        return [(meta_buffer_type, param_buffer_type)]

    def save_model(self, name, namespace):
        """
        Save Scaler param and meta
        Parameters
        ----------
        name: str, save DTable table_name
        namespace: str, DTable namespace

        Returns
        ----------
        tuple,
            include meta string name and param string name
        """
        if self.scale_param.method == consts.MINMAXSCALE:
            LOGGER.debug("save min_max scale model")
            return self.save_min_max_model(name, namespace)
        elif self.scale_param.method == consts.STANDARDSCALE:
            LOGGER.debug("save standard scale model")
            return self.save_standard_scale_model(name, namespace)
        else:
            LOGGER.debug("can not save {} model".format(self.scale_param.method))
            return None

    def load_model(self, name, namespace, header):
        """
        load Scaler model
        Parameters
        ----------
        name: str, DTable table_name
        namespace: str, DTable namespace

        Returns
        ----------
        list,
            scale information
        """
        self.header = header
        param_buffer_type = "{}.param".format(self.class_name)
        param_obj = feature_scale_param_pb2.ScaleParam()
        model_manager.read_model(buffer_type=param_buffer_type,
                                 proto_buffer=param_obj,
                                 name=name,
                                 namespace=namespace)

        if self.scale_param.method == consts.MINMAXSCALE:
            cols_scale_value = []
            param_dict = param_obj.minmax_scale_param

            scale_column_idx = []
            for idx, header_name in enumerate(self.header):
                feat_upper = 1
                feat_lower = 0
                out_lower = 0
                out_upper = 1

                if header_name in param_dict:
                    feat_upper = param_dict[header_name].feat_upper
                    feat_lower = param_dict[header_name].feat_lower
                    out_upper = param_dict[header_name].out_upper
                    out_lower = param_dict[header_name].out_lower
                    scale_column_idx.append(idx)
                else:
                    LOGGER.info(
                        "Can not find the header name {} in model.set some mask value to it".format(header_name))

                cols_scale_value.append((feat_lower,
                                         feat_upper,
                                         out_lower,
                                         out_upper))

            model_scale_value_results = [cols_scale_value, scale_column_idx]

        elif self.scale_param.method == consts.STANDARDSCALE:
            mean = []
            std = []
            scale_column_idx = []
            param_dict = param_obj.standard_scale_param
            for idx, header_name in enumerate(self.header):
                if header_name in param_dict:
                    mean.append(param_dict[header_name].mean)
                    std.append(param_dict[header_name].scale)
                    scale_column_idx.append(idx)
                else:
                    LOGGER.info(
                        "Can not find the header name {} in model.set some mask value to it".format(header_name))
                    mean.append(0)
                    std.append(1)

            model_scale_value_results = [mean, std, scale_column_idx]

        else:
            raise ValueError("Unknown scale method:{}".format(self.scale_param.method))

        return model_scale_value_results
