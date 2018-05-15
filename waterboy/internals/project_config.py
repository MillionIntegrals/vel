import os.path
import yaml


class ProjectConfig:
    """ Global configuration from the whole project taken from the .pbproject.yaml """
    PROJECT_FILE_NAME = '.wbproject.yaml'

    @staticmethod
    def find_project_directory(start_path) -> str:
        """ Locate top-level project directory  """
        start_path = os.path.realpath(start_path)
        possible_name = os.path.join(start_path, ProjectConfig.PROJECT_FILE_NAME)

        if os.path.exists(possible_name):
            return start_path
        else:
            up_path = os.path.realpath(os.path.join(start_path, '..'))
            if os.path.realpath(start_path) == up_path:
                raise RuntimeError("Couldn't find project file")
            else:
                return ProjectConfig.find_project_directory(up_path)

    def __init__(self, config_path):
        self.project_dir = ProjectConfig.find_project_directory(os.path.dirname(config_path))

        with open(os.path.join(self.project_dir, ProjectConfig.PROJECT_FILE_NAME), 'r') as fp:
            self.contents = yaml.safe_load(fp)

    def project_output_dir(self, *args):
        """ Directories where to store project files """
        return os.path.join(self.project_dir, 'output', *args)

    def tensorboard_dir(self, run_name) -> str:
        """ Directory where to store tensorboard files """
        return self.project_output_dir('tensorboard', run_name)

    def openai_dir(self, run_name) -> str:
        """ Directory where to store openai files """
        return self.project_output_dir('openai', run_name)
