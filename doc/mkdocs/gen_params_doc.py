"""Generate parms pages."""
import os
import mkdocs_gen_files

repo_base = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir
    )
)
params_source = os.path.join(repo_base, "python", "federatedml", "param")
params_doc_target = os.path.join(repo_base, "doc", "federatedml_component", "params")
md_template = """\
# {name}
::: federatedml.param.{name}
    options:
      heading_level: 2
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
"""


def create_params_doc():
    os.makedirs(params_doc_target, exist_ok=True)
    for file_name in os.listdir(params_source):
        if file_name.endswith(".py") and file_name != "__init__.py":
            name = file_name[:-3]
            full_doc_path = os.path.join(params_doc_target, f"{name}.md")
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                print(md_template.format(name=name), file=fd)
            mkdocs_gen_files.set_edit_path(full_doc_path, os.path.join(params_source, file_name))
