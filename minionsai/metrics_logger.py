"""
This library is trying to do the same thing as:
https://pypi.org/project/ml-logger/

But that one doesn't seem to work on windows

Usage:
```
from .metrics_logger import metrics_logger
metrics_logger.configure(os.path.join(someplace))
for step in range(100):
    metrics_logger.log_metrics({"loss": loss.item(), "step": step})
    metrics_logger.flush()
```
"""

from collections import defaultdict
from contextlib import contextmanager
import os
import time
import tabulate
import logging
import csv
import os
import numpy as np

logger = logging.getLogger(__name__)

class MetricsLogger():
    def __init__(self):
        self._csv_path = None
        self._metrics_this_step = {}
        self._seen_keys = []
        self._last_seen_values = {}
        self._timings = defaultdict(list)

    def configure(self, path):
        self._csv_path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self._seen_keys)

    @property
    def csv_path(self):
        if self._csv_path is None:
            raise Exception("Logger not configured. call logger.configure(path)")
        return self._csv_path

    def log_metrics(self, metrics, prefix=""):
        if len(prefix) > 0 and not prefix.endswith("/"):
            prefix = prefix + "/"
        for key, value in metrics.items():
            full_key = prefix + key
            if full_key in self._metrics_this_step:
                raise ValueError(f"Metric {full_key} already logged this step")
            self._metrics_this_step[full_key] = value

    @contextmanager
    def timing(self, key):
        """
        Contextmanager that adds the average time taken within it as a metric
        """
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self._timings[key].append(end - start)
    
    def format_if_number(self, number):
        """
        Prints a number in a pretty format
        With commas separating thousands if they're big enough
        expoenntial notation if small enough
        and a reasonable number of decimal places.
        """
        if isinstance(number, int):
            return f"{number:,}"
        if isinstance(number, float):
            if np.abs(number) < 1e-3:
                return f"{number:.4e}"
            else:
                return f"{number:.4f}"
        return number

    def _flush_timings(self):
        for key, timings in self._timings.items():
            if len(timings) > 1:
                self._metrics_this_step[f"timing/{key}/total"] = sum(timings)
                self._metrics_this_step[f"timing/{key}/avg"] = np.mean(timings)
                self._metrics_this_step[f"timing/{key}/std"] = np.std(timings)
                self._metrics_this_step[f"timing/{key}/n"] = len(timings)
            else:
                self._metrics_this_step[f"timing/{key}"] = timings[0]
        self._timings = defaultdict(list)

    def flush(self):
        self._flush_timings()
        new_keys = list(set(self._metrics_this_step.keys()) - set(self._seen_keys))
        if len(new_keys) > 0:
            logger.info(f"New metrics created this step: {new_keys}")
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == self._seen_keys
                old_data = list(reader)
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header + new_keys)
                writer.writerows([row + [""] * len(new_keys) for row in old_data])
            self._seen_keys = self._seen_keys + new_keys
        with open(self.csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self._metrics_this_step.get(key, "") for key in self._seen_keys])

        results = [(key, self._metrics_this_step[key]) if key in self._metrics_this_step else (key+"*", self._last_seen_values[key])
                    for key in self._seen_keys]
        results = [(k, self.format_if_number(v)) for k, v in results]
        str_table = tabulate.tabulate(sorted(results), tablefmt="pretty", colalign=("left", "left"))
        logger.info("\n" + str_table)

        self._last_seen_values.update(self._metrics_this_step)
        self._metrics_this_step = {}

metrics_logger = MetricsLogger()