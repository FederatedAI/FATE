import os
import shutil
import re
import glob

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


_INCLUDE_EXAMPLES_REGEX = re.compile(
    r"""(?P<_includer_indent>[^\S\r\n]*){\s*%\s*include-examples\s*"(?P<example_name>[^")]+)"\s*%\s*}\s*""",
    flags=re.VERBOSE | re.DOTALL,
)

_INCLUDE_EXAMPLE_REGEX = re.compile(
    r"""(?P<_includer_indent>[^\S\r\n]*){\s*%\s*include-example\s*"(?P<example_path>[^")]+)"\s*%\s*}\s*""",
    flags=re.VERBOSE | re.DOTALL,
)

_LINT_MAP = {
    ".py": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "sh",
    ".md": "md",
}

_EXAMPLES_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
)


def sub_include_examples(match):

    example_name = match.group("example_name")
    indents_level0 = match.group("_includer_indent")

    lines = []
    lines.append(f"{indents_level0}???+ Example\n")
    lines.append(f"{indents_level0}\n")
    indents_level1 = indents_level0 + "    "
    for example_type in ["pipeline", "dsl/v2", "dsl/v1"]:
        include_path = os.path.join(_EXAMPLES_BASE, example_type, example_name, "*.*")
        lines.append(f'{indents_level1}=== "{example_type}"\n\n')
        indents_level2 = f"{indents_level1}    "

        if example_type == "dsl/v1":
            lines.append(f"{indents_level2}!!! warning\n")
            lines.append(
                f"{indents_level2}    `dsl version 1` is deprecated, please consider use `dsl version v2` or `pipeline` instead!\n"
            )
            lines.append(f"{indents_level2}    \n")

        for name in glob.glob(include_path):
            if name.endswith("README.md") or name.endswith("readme.md"):
                lines.append(f"{indents_level2}```markdown\n")
                with open(name) as f:
                    for line in f.readlines():
                        lines.append(f"{indents_level2}{line}")

                lines.append(f"{indents_level2}\n")
                lines.append(f"{indents_level2}```\n")
                lines.append(f"{indents_level2}\n")

        for file_name in glob.glob(include_path):
            if file_name.endswith("README.md") or file_name.endswith("readme.md"):
                continue
            _, file_extension = os.path.splitext(file_name)
            lint = _LINT_MAP.get(file_extension, "")
            lines.append(
                f'{indents_level2}??? Example "{os.path.basename(file_name)}"\n'
            )
            lines.append(f"{indents_level2}    ```{lint}\n")
            head = True
            with open(file_name) as f:
                for line in f.readlines():
                    # skip license
                    if head:
                        if line.strip() == "" or line.lstrip().startswith("#"):
                            continue
                    head = False
                    lines.append(f"{indents_level2}    {line}")
            lines.append(f"{indents_level2}    \n")
            lines.append(f"{indents_level2}    ```\n")
            lines.append(f"{indents_level2}    \n")

    return "".join(lines)


def sub_include_example(src_file_path):
    def sub(match):
        example_path = match.group("example_path")
        indents_level0 = match.group("_includer_indent")

        lines = []
        lines.append(f"{indents_level0}\n")
        lines.append(f'{indents_level0}??? Example "{example_path}"\n')
        lines.append(f"{indents_level0}\n")
        indents_level1 = indents_level0 + "    "
        abs_file_path = os.path.abspath(
            os.path.join(src_file_path, os.pardir, example_path)
        )
        if os.path.exists(abs_file_path):
            with open(abs_file_path) as f:
                _, file_extension = os.path.splitext(abs_file_path)
                lint = _LINT_MAP.get(file_extension, "")
                lines.append(f"{indents_level1}```{lint}\n")
                head = True
                for line in f.readlines():
                   # skip license
                    if head:
                      if line.strip() == "" or line.lstrip().startswith("#"):
                         continue
                    head = False
                    lines.append(f"{indents_level1}    {line}")
                lines.append(f"{indents_level1}\n")
                lines.append(f"{indents_level1}```\n")
                lines.append(f"{indents_level1}\n")
        
        return "".join(lines)

    return sub


_MARKDOWN_URL_REGEX = re.compile(
    r"""(?P<text>\[[^\(]?\])\((?P<url>[^\)]+)\)""",
    flags=re.VERBOSE | re.DOTALL,
)


def _fix_zh_url(match):
    text = match.group("text")
    url = match.group("url")

    if not url.startswith("http"):
        url = os.path.join(os.pardir, url)
    return f'{text}({url})'



_COMMENT_REGEX = re.compile(
    r"""[^\S\r\n]*<!--\s*mkdocs\s*\n(?P<_content>.*?)-->""",
    flags=re.VERBOSE | re.DOTALL,
)

def _remove_comment(match):
    content = match.group("_content")
    return content

def on_page_markdown(markdown, page, **kwargs):
    if page.file.abs_src_path.rsplit(".", 2)[-2] == "zh":
        markdown = re.sub(_MARKDOWN_URL_REGEX, _fix_zh_url, markdown)

    # remove specific commnent    
    markdown = re.sub(
        _COMMENT_REGEX,
        _remove_comment,
        markdown
    )

    markdown = re.sub(
        _INCLUDE_EXAMPLES_REGEX,
        sub_include_examples,
        markdown,
    )

    markdown = re.sub(
        _INCLUDE_EXAMPLE_REGEX,
        sub_include_example(page.file.abs_src_path),
        markdown,
    )
    return markdown

