import argparse
import importlib
import json
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
config_yaml_file = os.path.join(cur_dir, '../config.yaml')
TEMP_FILE_PATH = os.path.join(cur_dir, 'temp_pipeline.py')


class ConfDSLGenerator(object):
    def __make_temp_pipeline(self, pipeline_file):
        code_list = []
        with open(pipeline_file, 'r') as f:
            lines = f.readlines()
            start_main = False
            has_returned = False
            space_num = 0
            for line in lines:
                if "def main" in line:
                    for char in line:
                        if char.isspace():
                            space_num += 1
                        else:
                            break
                    start_main = True
                    code_list.append(line)

                elif start_main and "def " in line and not has_returned:
                    code_list.append(" " * (space_num + 4) + "return pipeline\n")
                    start_main = False
                    code_list.append(line)

                elif start_main and "return " in line:
                    code_list.append(" " * (space_num + 4) + "return pipeline\n")
                    start_main = False

                elif 'if __name__ ==' in line:
                    code_list.append(" " * (space_num + 4) + "return pipeline\n")
                    start_main = False
                    code_list.append(line)

                else:
                    code_list.append(line)

            if start_main:
                code_list.append(" " * (space_num + 4) + "return pipeline\n")

        with open(TEMP_FILE_PATH, 'w') as f:
            f.writelines(code_list)

    def __run_pipeline(self):
        loader = importlib.machinery.SourceFileLoader("main", str(TEMP_FILE_PATH))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        pipeline = mod.main(config_yaml_file)

        return pipeline

    def convert(self, pipeline_file):
        self.__make_temp_pipeline(pipeline_file)
        my_pipeline = self.__run_pipeline()
        conf = my_pipeline.get_train_conf()
        dsl = my_pipeline.get_train_dsl()

        folder_name, file_name = os.path.split(pipeline_file)
        print(f"folder_name: {folder_name}, file_name: {file_name}")
        conf_name = file_name.replace('.py', '_conf.json')
        dsl_name = file_name.replace('.py', '_dsl.json')
        conf_name = os.path.join(folder_name, conf_name)
        dsl_name = os.path.join(folder_name, dsl_name)

        with open(conf_name, 'w') as f:
            json.dump(conf, f, indent=4)
            print('conf name is {}'.format(conf_name))
        with open(dsl_name, 'w') as f:
            json.dump(dsl, f, indent=4)
            print('dsl name is {}'.format(dsl_name))

        self.__delete_temp_file()

    def __delete_temp_file(self):
        os.remove(TEMP_FILE_PATH)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-c", "--config_file", type=str, help="config file", required=True)

    args = arg_parser.parse_args()
    config_file = args.config_file

    conf_generator = ConfDSLGenerator()
    conf_generator.convert(config_file)
