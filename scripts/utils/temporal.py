"""Time utils.
"""

import re
import time
import logging
from pathlib import Path
from datetime import datetime


def get_current_timestamp():
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")


class TimeIt():
    """Class used to time code execution and write results to a file (and logs).

    Usage example:
    >>> with TimeIt(path_to_file):
    >>>     time.sleep(1)
    >>> # This will write the wall-clock and process time of the sleep call to
    >>> # the file at `path_to_file`
    """

    def __init__(self, output_path: str | Path = None, name: str = None):
        self.output_path = output_path
        self.name = name
        self.start_wall_time = None
        self.start_process_time = None

    def __enter__(self):
        self.start_wall_time = time.time()
        self.start_process_time = time.process_time()
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        """Exit the runtime context and return a boolean indicating whether any
        exceptions that occurred should be suppressed.

        If an exception occurred while executing the `with` body, then the three
        arguments to this function will be filled with the exception's type, 
        value, and traceback data (otherwise they will all be Null).
        """

        elapsed_process_time = time.process_time() - self.start_process_time
        elapsed_wall_time = time.time() - self.start_wall_time

        # Log time results
        msg = f"\n> {self.name} ::\n" if self.name else ""
        msg += (
            f"Elapsed wall-clock time: {elapsed_wall_time:.3f}\n"
            f"Elapsed process time: {elapsed_process_time:.3f}\n"
        )
        logging.info(msg)

        # And print to disk (optional)
        if self.output_path is not None:
            with open(self.output_path, "w") as f_out:
                print(msg, file=f_out)
        
        # Don't suppress any occurred exceptions
        return False
    
    @staticmethod
    def parse_time_file(path: Path) -> dict:
        """Parses a file with `TimeIt` information.
        """
        wallclock_regex = r"wall-clock time: (?P<time>\d+([.]\d*)?)"
        processtime_regex = r"process time: (?P<time>\d+([.]\d*)?)"
        
        with open(path, "r") as f_in:
            whole_file = f_in.read()

        wc_match = re.search(wallclock_regex, whole_file)
        pc_match = re.search(processtime_regex, whole_file)

        if not wc_match or not pc_match:
            logging.error(f"Failed to match contents of `TimeIt` file at {path}")
            return {}

        return {
            "wall-clock_time": float(wc_match.group("time")),
            "process_time": float(pc_match.group("time")),
        }

