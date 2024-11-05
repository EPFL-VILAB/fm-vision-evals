import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import argparse
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from taskit.mfm import get_mfm_wrapper
from taskit.eval import eval_classify, eval_segment, eval_grouping, eval_depth, eval_normals, eval_object
from taskit.tasks import classify, segment, grouping, depth, normals, object
from scripts.utils.log import Logger

file_lock = threading.Lock()


def get_args():
    parser = argparse.ArgumentParser('inferring CV tasks using multimodal LLMs', add_help=True)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser.add_argument('--model',
                        choices=['gpt-4o-2024-08-06', 'gemini-1.5-pro', 'claude-3-5-sonnet-20240620'],
                        type=str, required=True, help='MFM API to use')
    parser.add_argument('--api_key', type=str, required=True, help='API key for the MFM API')

    parser.add_argument('-t', '--task', default='', type=str,
                        choices=['classify', 'segment', 'group', 'depth', 'normals', 'detect'],
                        help='Task to evaluate MFM on')
    parser.add_argument('-e', '--eval_type', default='', type=str,
                        help='Type of eval to run')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Number of files to process in a batch.')
    parser.add_argument('--n_threads', default=1, type=int,
                        help='Number of threads to use for processing')

    parser.add_argument('-d', '--data_files', type=str,
                        help='Path where the data file paths are stored')
    parser.add_argument('-l', '--log_name', default='output', type=str,
                        help='Name of the log files')
    parser.add_argument('--output_dir', default='./scripts/data/logs/outputs', type=str,
                        help='Path to store the output logs')
    parser.add_argument('--backup_dir', default='./scripts/data/logs/backups', type=str,
                        help='Path to store the backup logs')
    parser.add_argument('--error_dir', default='./scripts/data/logs/error_file_logs', type=str,
                        help='Path to store the files that caused errors')
    parser.add_argument('--eval_output_file', default='', type=str,
                        help="File path of output to run eval on")

    parser.add_argument('--only_eval', action='store_true',
                        help='If only evaluation is to be run.')
    parser.add_argument('--no_eval', action='store_true',
                        help='If evaluation is to be skipped.')
    parser.add_argument('--continue', action='store_true',
                        help='If the evaluation is to be continued from the last checkpoint.')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Overwrite default arguments with the ones from the config file (except 'task_specific_args')
            for key, value in config.items():
                if key != 'task_specific_args' and key != 'eval_specific_args':
                    setattr(args, key, value)

    if args.config:
        return args, config['task_specific_args'], config['eval_specific_args']
    else:
        return args, {}, {}

def find_points_for_grouping(f):
    groundtruth = json.load(open('./taskit/utils/metadata/coco-group.json'))[f]
    points = [v['point'] for k, v in groundtruth.items() if k != 'file_name']
    return points


def main(args, task_specific_args, eval_specific_args):

    log = Logger(path=args.output_dir, backup_path=args.backup_dir, error_path=args.error_dir, output_file_name=args.log_name)
    model = get_mfm_wrapper(args.model, args.api_key)

    # Load the data files
    with open(args.data_files) as f:
        data_files = [x.strip() for x in f.readlines()]
        # If there is a batch_size, group the files into batches
        if args.batch_size > 1:
            data_files = [data_files[i:i + args.batch_size] for i in range(0, len(data_files), args.batch_size)]

    if not args.only_eval:
        log.log_info({"task": "args.task"})
        log.info("Beginning inference")

        def process_iter(index, f):
            if args.task == 'group':
                point_list = find_points_for_grouping(f)
                task_specific_args['point_list'] = point_list
            resp_dict, tokens, error_status = model.predict(args.task, f, return_dict=True, **task_specific_args)
            if error_status:
                log.log_invalid_file(f)
                log.error(f"Error in processing {f}")
                return index, None, tokens
            else:
                return index, resp_dict, tokens

        compl_tokens, prompt_tokens = 0, 0
        with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            futures = {executor.submit(process_iter, index, f): (index, f) for index, f in enumerate(data_files)}
            results = [None] * len(data_files)

            for future in (pbar := tqdm(as_completed(futures), total=len(futures))):
                index, resp_dict, tokens = future.result()
                if resp_dict is not None and tokens is not None:
                    compl_tokens, prompt_tokens = compl_tokens + tokens[0], prompt_tokens + tokens[1]
                    results[index] = resp_dict, tokens

                    log.log_backup(resp_dict)

                pbar.set_description(f"Completion tokens: {compl_tokens} | Prompt tokens: {prompt_tokens}")

        for result in results:
            if result is not None:
                resp_dict, tokens = result
                log.log_output(resp_dict, tokens)

        log.info("Inference complete")
        log.log_all()

    if not args.no_eval:
        log.info("Beginning evaluation")
        output_file = args.eval_output_file if args.only_eval else log.get_output_file()
        if args.eval_type:
            eval_metric = model.eval(eval=args.eval_type, predictions=output_file, invalid_files=log.get_invalid_files(), **eval_specific_args)
        else:
            eval_metric = model.eval(eval=None, predictions=output_file, task=args.task, invalid_files=log.get_invalid_files())
        log.info("Evaluation complete")
        log.log_update({"eval_metric": str(eval_metric)})


if __name__ == '__main__':
    args, task_specific_args, eval_specific_args = get_args()
    main(args, task_specific_args, eval_specific_args)
