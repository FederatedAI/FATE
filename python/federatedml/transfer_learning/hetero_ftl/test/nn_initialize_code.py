# np.random.seed(100)
# if self.role == consts.HOST:
#     np_weight_w = np.random.normal(size=(10, self.nn._model.output_shape[1]))
#     np_weight_w = np.random.normal(size=(self.nn._model.input_shape[1], self.nn._model.output_shape[1]))
# else:
#     np_weight_w = np.random.normal(size=(self.nn._model.input_shape[1], self.nn._model.output_shape[1]))
#
# np_weight_b = np.zeros((self.nn._model.output_shape[1], ))
# LOGGER.debug('weights are {}, shape is {}'.format(np_weight_w, np_weight_w.shape))
#
# self.nn._model.set_weights([np_weight_w, np_weight_b])