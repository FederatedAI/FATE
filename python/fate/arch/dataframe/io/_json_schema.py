FRAME_SCHEME = "fate.dataframe"


def build_schema(data, global_ranks):
    fields = []
    schema = data.schema
    """
    index, match_id, label, weight, values
    """
    fields.append(
        dict(
            type="str",
            name=schema["sid"],
            property="index"
        )
    )

    if schema.get("match_id_name") is not None:
        fields.append(
            dict(
                type="str",
                name=schema["match_id"],
                property="match_id"
            )
        )

    if schema.get("label_name") is not None:
        label = data.label
        fields.append(
            dict(
                type=label.dtype.name,
                name=schema["label_name"],
                property="label"
            )
        )

    if schema.get("weight_name") is not None:
        weight = data.weight
        fields.append(
            dict(
                type=weight.dtype.name,
                name=schema["weight_name"],
                property="weight"
            )
        )

    if schema.get("header") is not None:
        values = data.values
        columns = schema.get("header")
        for col_name in columns:
            fields.append(
                dict(
                    type=values.dtype.name,
                    name=col_name,
                    property="value"
                )
            )

    schema["fields"] = fields
    schema["global_ranks"] = global_ranks
    schema["type"] = FRAME_SCHEME
    return schema


def parse_schema(schema):
    if "type" not in schema or schema["type"] != FRAME_SCHEME:
        raise ValueError(f"deserialize data error, schema type is not {FRAME_SCHEME}")

    recovery_schema = dict()
    column_info = dict()
    fields = schema["fields"]

    for idx, field in enumerate(fields):
        if field["property"] == "index":
            recovery_schema["sid"] = field["name"]
            column_info["index"] = dict(start_idx=idx,
                                        end_idx=idx,
                                        type=field["type"])

        elif field["property"] == "match_id":
            recovery_schema["match_id_name"] = field["name"]
            column_info["match_id"] = dict(start_idx=idx,
                                           end_idx=idx,
                                           type=field["type"])

        elif field["property"] == "label":
            recovery_schema["label_name"] = field["name"]
            column_info["label"] = dict(start_idx=idx,
                                        end_idx=idx,
                                        type=field["type"])

        elif field["property"] == "weight":
            recovery_schema["weight_name"] = field["name"]
            column_info["weight"] = dict(start_idx=idx,
                                         end_idx=idx,
                                         type=field["type"])

        elif field["property"] == "value":
            header = [field["name"] for field in fields[idx:]]
            recovery_schema["header"] = header
            column_info["values"] = dict(start_idx=idx,
                                         end_idx=idx + len(header) - 1,
                                         type=field["type"])
            break

    return recovery_schema, schema["global_ranks"], column_info
