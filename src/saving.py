import json
import os
from typing import Optional, Any, Union

from context_printer import ContextPrinter as Ctp


def dumper(obj: Any) -> Union[dict, str]:
    if isinstance(obj, type):
        return obj.__name__
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__


def save_results(path: str, local_results: dict, new_devices_results: dict, params_dict: dict) -> None:
    # Save the results to a new unique file (file name based on current time)
    with open(path + 'local_results.json', 'w') as outfile:
        json.dump(local_results, outfile, default=dumper, indent=2)
    with open(path + 'new_devices_results.json', 'w') as outfile:
        json.dump(new_devices_results, outfile, default=dumper, indent=2)
    with open(path + 'params_dict.json', 'w') as outfile:
        json.dump(params_dict, outfile, default=dumper, indent=2)


def save_results_gs(path: str, local_results: dict, params_dict: dict) -> None:
    # Save the results to a new unique file (file name based on current time)
    with open(path + 'local_results.json', 'w') as outfile:
        json.dump(local_results, outfile, default=dumper, indent=2)
    with open(path + 'params_dict.json', 'w') as outfile:
        json.dump(params_dict, outfile, default=dumper, indent=2)


def create_new_numbered_dir(base_path: str) -> Optional[str]:
    for run_id in range(1000):
        path = base_path + repr(run_id) + '/'
        if not os.path.exists(path):
            Ctp.print('Creating folder ' + path)
            os.makedirs(path)
            return path
    return None