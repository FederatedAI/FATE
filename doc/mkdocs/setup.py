import glob
import os
import re
import shutil

from setuptools import setup

repo_base = os.path.abspath(
    os.path.join(
        os.path.abspath(
            __file__), os.path.pardir, os.path.pardir, os.path.pardir
    )
)
params_source = os.path.join(repo_base, "python", "federatedml", "param")
params_doc_target = os.path.join(
    repo_base, "doc", "federatedml_component", "params")
md_template = """\
# {name}
::: federatedml.param.{name}
    rendering:
      heading_level: 2
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
"""


def pre_build(*args, **kwargs):
    clean_params_doc()
    create_params_doc()


def create_params_doc():
    os.makedirs(params_doc_target, exist_ok=True)
    for file_name in os.listdir(params_source):
        if file_name.endswith(".py") and file_name != "__init__.py":
            name = file_name[:-3]
            with open(os.path.join(params_doc_target, f"{name}.md"), "w") as f:
                f.write(md_template.format(name=name))


def clean_params_doc():
    try:
        shutil.rmtree(params_doc_target)
    except Exception as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


pre_build()
setup(
    name='fate-doc',
    version='1.0',
    description='fake module to generate docs',
    author='FederatedAI',
    packages=[],  # same as name
)
