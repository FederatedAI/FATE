import os

repo_base = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

def define_env(env): 
  @env.macro
  def include_source_code(path, num_space=0):
    padding = " " * num_space
    path = os.path.join(repo_base, path)
    with open(path) as f:
      lines = f.readlines()
      return "".join(f"{padding}{line}" if i > 0 else line for i, line in enumerate(lines))