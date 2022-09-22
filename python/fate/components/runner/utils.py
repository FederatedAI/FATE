def set_predict_data_schema(predict_datas, schemas):
    if predict_datas is None:
        return predict_datas
    if isinstance(predict_datas, list):
        predict_data = predict_datas[0]
        schema = schemas[0]
    else:
        predict_data = predict_datas
        schema = schemas
    if predict_data is not None:
        predict_data.schema = {
            "header": [
                "label",
                "predict_result",
                "predict_score",
                "predict_detail",
                "type",
            ],
            "sid": schema.get("sid"),
            "content_type": "predict_result",
        }
        if schema.get("match_id_name") is not None:
            predict_data.schema["match_id_name"] = schema.get("match_id_name")
    return predict_data


def union_data(previews_data, name_list):
    if len(previews_data) == 0:
        return None

    if any([x is None for x in previews_data]):
        return None

    assert len(previews_data) == len(name_list)

    def _append_name(value, name):
        inst = copy.deepcopy(value)
        if isinstance(inst.features, list):
            inst.features.append(name)
        else:
            inst.features = np.append(inst.features, name)
        return inst

    result_data = None
    for data, name in zip(previews_data, name_list):
        # LOGGER.debug("before mapValues, one data: {}".format(data.first()))
        f = functools.partial(_append_name, name=name)
        data = data.mapValues(f)
        # LOGGER.debug("after mapValues, one data: {}".format(data.first()))

        if result_data is None:
            result_data = data
        else:
            LOGGER.debug(
                f"Before union, t1 count: {result_data.count()}, t2 count: {data.count()}"
            )
            result_data = result_data.union(data)
            LOGGER.debug(f"After union, result count: {result_data.count()}")
        # LOGGER.debug("before out loop, one data: {}".format(result_data.first()))

    return result_data
