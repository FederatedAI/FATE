import sys
import yaml


def gen_code(component_dict):
    module = component_dict.get("module")
    input_data_key = component_dict.get("input_data_key")
    input_model_key = component_dict.get("input_model_key")
    if input_data_key is not None:
        if isinstance(input_data_key, list):
            input_data_key = "[" + ", ".join(input_data_key) + "]"

    if input_model_key is not None:
        if isinstance(input_model_key, list):
            input_model_key = "[" + ", ".join(input_model_key) + "]"

    output_key = component_dict.get("output")
    if output_key:
        output_key = list(output_key.keys())
        output_key = "[" + ", ".join(output_key) + "]"

    output_str = ""
    output_str += "from pipeline.components import ComponentBase\n"
    output_str += "from pipeline.interface import Input, Output\n\n\n"
    output_str += f"class {module}(ComponentBase):\n"
    output_str += f"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module = {module}
        self._input = Input(self.name, data_key={input_data_key}, model_key={input_model_key})
        self.output = Output(self.name, output_key={output_key})
    """
    return output_str


if __name__ == "__main__":
    yaml_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(yaml_file, "r") as fin:
        buf = yaml.load(fin.read())

    if not buf:
        raise ValueError("Can not load yaml file")

    out_buf = gen_code(buf)
    with open(out_file, "w") as fout:
        fout.write(out_buf)
