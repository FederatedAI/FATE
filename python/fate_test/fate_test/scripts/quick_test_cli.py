import os
import subprocess
import click
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions


@click.group(name="quick-test")
def quick_test_group():
    """
    quick test
    """
    ...


@quick_test_group.command("federatedml")
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
    unit_test_path = "./python/federatedml/"
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    run_test(includes=include, conf=config_inst, error_log_file=error_log_file, unit_test_path=unit_test_path)


def run_test(includes, conf: Config, error_log_file, unit_test_path):
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

    failed_count = 0
    ml_dir = conf.federatedml_dir
    PYTHONPATH = os.path.join(conf.data_base_dir, ml_dir.split("federatedml")[0])
    os.environ['PYTHONPATH'] = PYTHONPATH
    if len(includes) == 0:
        ml_dir = conf.federatedml_dir
        traverse_folder(ml_dir)
    else:
        ml_dir = includes
        for v in ml_dir:
            traverse_folder(os.path.abspath(v))

    echo.echo(f"there are {failed_count} failed test")
    print('Please check the error content: {}'.format(error_log(None)))
