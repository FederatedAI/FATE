import os
import re
import glob
import git

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

_REPO_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

_EXAMPLES_BASE = os.path.abspath(os.path.join(_REPO_BASE, "examples"))


def sub_include_examples(match):

    example_name = match.group("example_name")
    indents_level0 = match.group("_includer_indent")

    lines = []
    lines.append(f"{indents_level0}???+ Example\n")
    lines.append(f"{indents_level0}\n")
    indents_level1 = indents_level0 + "    "
    for example_type, pretty_name in [("pipeline", "Pipeline"), ("dsl/v2", "DSL")]:
        include_path = os.path.join(_EXAMPLES_BASE, example_type, example_name, "*.*")
        lines.append(f'{indents_level1}=== "{pretty_name}"\n\n')
        indents_level2 = f"{indents_level1}    "

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


try:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    url = repo.remote().url
    if url.endswith(".git"):
        url = url[:-4]
    GITHUB_REPO = f"{url}/tree/{sha}"
except BaseException:
    GITHUB_REPO = "https://github.com/FederatedAI/FATE/tree/master"

_DIR_URL_REGEX = re.compile(
    r"""(?P<text>\[\s*:file_folder:[^\(]*\])\((?P<url>[^\)]+)\)""",
    flags=re.VERBOSE | re.DOTALL,
)


def _fix_dir_url(src_path):
    def _replace(match):
        text = match.group("text")
        url = match.group("url")

        if not url.startswith("http"):
            url_rel_to_repo_base = os.path.relpath(
                os.path.abspath(os.path.join(src_path, os.path.pardir, url)), _REPO_BASE
            )
            url = f"{GITHUB_REPO}/{url_rel_to_repo_base}"
        return f"{text}({url})"

    return _replace


_COMMENT_REGEX = re.compile(
    r"""[^\S\r\n]*<!--\s*mkdocs\s*\n(?P<_content>.*?)-->""",
    flags=re.VERBOSE | re.DOTALL,
)


def _remove_comment(match):
    content = match.group("_content")
    return content


def on_page_markdown(markdown, page, **kwargs):

    markdown = re.sub(_DIR_URL_REGEX, _fix_dir_url(page.file.abs_src_path), markdown)

    # remove specific commnent
    markdown = re.sub(_COMMENT_REGEX, _remove_comment, markdown)

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
