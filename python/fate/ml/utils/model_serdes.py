from typing import Dict, Tuple


def serialize_models(models):
    from google.protobuf import json_format

    serialized_models: Dict[str, Tuple[str, bytes, dict]] = {}

    for model_name, buffer_object in models.items():
        serialized_string = buffer_object.SerializeToString()
        pb_name = type(buffer_object).__name__
        json_format_dict = json_format.MessageToDict(
            buffer_object, including_default_value_fields=True
        )

        serialized_models[model_name] = (
            pb_name,
            serialized_string,
            json_format_dict,
        )

    return serialized_models
