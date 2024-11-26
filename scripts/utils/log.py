import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union


class Logger:
    def __init__(
        self,
        path: str,
        backup_path: str,
        error_path: str,
        output_file_name: str,
        info_dict: Dict = {},
        delete_temp: bool = True,
        level: int = logging.INFO,
        name: str = 'logger',
    ):
        """Initialize the logger with a name, log file, and logging level.

        Args:
            path (str): Path that the log files should belong to
            output_file_name (str): Name of the log files. Date and time will be appended
            delete_temp (bool): Whether the temporary log file should be deleted after a run. Defaults to True
            level (int): The logging level. Defaults to logging.INFO.
            name (str): The name of the logger.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(path):
            os.makedirs(path)
        self.output_file = os.path.join(path, f"{output_file_name}_{timestamp}.json")
        self.temp_output_file = os.path.join(path, f"temp_{output_file_name}_{timestamp}.txt")
        self.backup_file = os.path.join(backup_path, f"backup_{output_file_name}_{timestamp}.json")
        self.error_file = os.path.join(error_path, f"error_{output_file_name}_{timestamp}.json")

        self.delete_temp = delete_temp
        self.info_dict = info_dict
        self.compl_tokens, self.prompt_tokens = 0, 0

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.invalid_files = []

        # Create console handler to log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(console_handler)

    # --custom logging methods----------------------------------------------------------------

    def log_output(self, resp_dict: Optional[Union[Dict, List]], tokens: Tuple[int, int]):
        """ Logs a single output of the API call"""
        self.compl_tokens += tokens[0]
        self.prompt_tokens += tokens[1]
        if resp_dict:
            with open(self.temp_output_file, 'a') as f:
                if isinstance(resp_dict, list):
                    for rd in resp_dict:
                        f.write(json.dumps(rd) + '\n')
                else:
                    f.write(json.dumps(resp_dict) + '\n')

    def log_backup(self, resp_dict: Union[Dict, List[Dict]]):
        """ Logs a backup of the output(s)"""
        if isinstance(resp_dict, dict):
            with open(self.backup_file, 'a') as f:
                f.write(json.dumps(resp_dict) + '\n')

        elif isinstance(resp_dict, list):
            with open(self.backup_file, 'a') as f:
                for resp in resp_dict:
                    f.write(json.dumps(resp) + '\n')

    def log_invalid_file(self, file_name: Union[str, List[str]]):
        """ Logs a file(s) that was not processed correctly"""
        if isinstance(file_name, str):
            with open(self.error_file, 'a') as f:
                f.write(file_name.strip() + '\n')
        elif isinstance(file_name, list):
            with open(self.error_file, 'a') as f:
                for fn in file_name:
                    f.write(fn.strip() + '\n')

    def log_all(self):
        """ After a run is over, writes token info and transfers everything in the temp log file to
        the log file in a JSON format"""
        with open(self.output_file, 'w') as f:
            f.write(json.dumps({
                "completion_tokens": self.compl_tokens,
                "prompt_tokens": self.prompt_tokens,
                **self.info_dict,
                "data": []
            })[:-2] + '\n')

            # check if temp_output_file exists
            if os.path.exists(self.temp_output_file):
                with open(self.temp_output_file, 'r') as temp_f:
                    data_lines = [line.strip() for line in temp_f if line.strip()]
                    data_json = ',\n'.join(data_lines)

                f.write(f"{data_json}\n]}}")

                if self.delete_temp:
                    os.remove(self.temp_output_file)
            else:
                f.write("]}")

    def log_info(self, new_dict: Dict):
        """ Adds more info to info_dict"""
        self.info_dict.update(new_dict)

    def log_update(self, new_dict: Dict):
        """ Adds new info to output_file"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data.update(new_dict)

        with open(self.output_file, 'w') as f:
            f.write(json.dumps(data))

    def get_invalid_files(self):
        return self.invalid_files

    # --generic logging methods----------------------------------------------------------------

    def debug(self, message: str):
        """Log a message with level DEBUG."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log a message with level INFO."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a message with level WARNING."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log a message with level ERROR."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log a message with level CRITICAL."""
        self.logger.critical(message)

    # --getters----------------------------------------------------------------

    def get_output_file(self):
        return self.output_file


# Example usage
if __name__ == "__main__":
    log = Logger(path=".", backup_path=".", error_path=".", output_file_name="test", info_dict={"name": "test"}, delete_temp=False)
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
    log.log_output({"test": "123"}, (5, 42))
    log.log_output({"test": "456"}, (5, 42))
    log.log_all()
