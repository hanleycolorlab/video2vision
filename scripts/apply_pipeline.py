import argparse

from tabulate import tabulate

import video2vision as v2v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_path', type=str,
                        help='Path to serialized pipeline')
    parser.add_argument('paths', type=str, nargs='+',
                        help='Paths to attach Loaders and Writers to')
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    pipe = v2v.load_pipeline(args.pipeline_path)
    pipe.set_all_paths(*args.paths)
    if args.batch_size is not None:
        pipe.set_batch_size(args.batch_size)
    rtn = pipe.run()

    print(tabulate(rtn.items()))
