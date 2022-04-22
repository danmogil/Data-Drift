import datetime
from pathlib import Path

class FileManager:
    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.time = str(datetime.datetime.now())
        self.modified_input_path = f"{self.input_path}/{self.time}"
        self.modified_output_path = f"{self.output_path}/{self.time}"

    def _prepare_input_path(self):
        p = Path(self.modified_input_path)
        p.mkdir(exist_ok=True)

    def _prepare_output_path(self):
        p = Path(self.modified_output_path)
        p.mkdir(exist_ok=True)

    def get_modified_input_path(self):
        self.modified_input_path()
        return self.input_path

    def get_modified_output_path(self):
        self._prepare_output_path()
        return self.modified_output_path