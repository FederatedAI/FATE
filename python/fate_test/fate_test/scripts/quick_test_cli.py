import os
import subprocess
import click
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions


@click.group(name="unittest")
def unittest_group():
    """
    unit test
    """
    ...


@unittest_group.command("federatedml")
@click.option('-i', '--include', type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="Specify federatedml test units for testing")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def unit_test(ctx, include, **kwargs):
    """
    federatedml unit test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.echo(f"testsuite namespace: {namespace}", fg='red')

    if not yes and not click.confirm("running?"):
        return

    error_log_file = f"./logs/{namespace}/error_test.log"
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    run_test(includes=include, conf=config_inst, error_log_file=error_log_file)


def run_test(includes, conf: Config, error_log_file):
    def error_log(stdout):
        if stdout is None:
            return os.path.abspath(error_log_file)
        with open(error_log_file, "a") as f:
            f.write(stdout)

    def run_test(file):
        global failed_count
        echo.echo("start to run test {}".format(file))
        try:
            subp = subprocess.Popen(["python", file],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = subp.communicate()
            stdout = stdout.decode("utf-8")
            echo.echo(stdout)
            if "FAILED" in stdout:
                failed_count += 1
                error_log(stdout=f"error sequence {failed_count}: {file}")
                error_log(stdout=stdout)
        except Exception:
            return

    def traverse_folder(file_fullname):
        if os.path.isfile(file_fullname):
            if "_test.py" in file_fullname and "ftl" not in file_fullname:
                run_test(file_fullname)
        else:
            for file in os.listdir(file_fullname):
                file_fullname_new = os.path.join(file_fullname, file)
                if os.path.isdir(file_fullname_new):
                    traverse_folder(file_fullname_new)
                if "_test.py" in file and ("/test" in file_fullname or "tests" in file_fullname):
                    if "ftl" in file_fullname_new:
                        continue
                    else:
                        run_test(file_fullname_new)

    global failed_count
    failed_count = 0
    fate_base = conf.fate_base
    ml_dir = os.path.join(fate_base, "python/federatedml")
    PYTHONPATH = os.environ.get('PYTHONPATH') + ":" + os.path.join(fate_base, "python")
    os.environ['PYTHONPATH'] = PYTHONPATH
    if len(includes) == 0:
        traverse_folder(ml_dir)
    else:
        ml_dir = includes
        for v in ml_dir:
            traverse_folder(os.path.abspath(v))

    echo.echo(f"there are {failed_count} failed test")
    if failed_count > 0:
        print('Please check the error content: {}'.format(error_log(None)))
