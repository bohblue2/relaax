import os
import errno
import click
import shutil

from ..cmdl import pass_context


class NewApp(object):

    def __init__(self, ctx, app_name, environment):
        self.ctx = ctx
        self.app_name = app_name
        self.base_env = environment

    def mk_folder(self, name):
        try:
            os.makedirs(name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Can\'t create \'%s\'. Folder already exists.' % name)
            raise

    def mk_environment(self):
        app_path = os.path.abspath(os.path.join(os.getcwd(), self.app_name))
        module_path = os.path.dirname(os.path.abspath(__file__))
        template = '../../templates/environments/%s' % self.base_env
        template_path = os.path.abspath(os.path.join(module_path, template))
        try:
            shutil.copytree(template_path, app_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Can\'t create \'%s\'. Folder already exists.' % self.app_name)
            raise

    def create(self):
        try:
            self.mk_environment()
            self.ctx.log('Created application \'%s\' based on \'%s\' environment',
                         self.app_name, self.base_env)
            self.ctx.log('To run application, do: cd %s && relaax run' % self.app_name)
        except Exception as e:
            self.ctx.log('%s', str(e))


@click.command('new', short_help='Create new RELAAX application.')
@click.argument('app-name', required=True, type=click.STRING)
@click.option('--environment', '-e', default='basic', show_default=True,
              type=click.Choice(['basic', 'openai-gym', 'deepmind-lab']),
              help='Environment to base application on.')
@pass_context
def cmdl(ctx, app_name, environment):
    """Build new RELAAX application."""
    ctx.setup_logger()
    NewApp(ctx, app_name, environment).create()
