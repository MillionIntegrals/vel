#!/usr/bin/env python
import argparse
import multiprocessing
import sys

from vel.internals.model_config import ModelConfig
from vel.internals.parser import Parser


def main():
    """ Paperboy entry point - parse the arguments and run a command """
    parser = argparse.ArgumentParser(description='Paperboy deep learning launcher')

    parser.add_argument('config', metavar='FILENAME', help='Configuration file for the run')
    parser.add_argument('command', metavar='COMMAND', help='A command to run')
    parser.add_argument('varargs', nargs='*', metavar='VARARGS', help='Extra options to the command')
    parser.add_argument('-r', '--run_number', type=int, default=0, help="A run number")
    parser.add_argument('-d', '--device', default='cuda', help="A device to run the model on")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random seed for the project")
    parser.add_argument(
        '-p', '--param', type=str, metavar='NAME=VALUE', action='append', default=[],
        help="Configuration parameters"
    )
    parser.add_argument(
        '--continue', action='store_true', default=False, help="Continue previously started learning process"
    )
    parser.add_argument(
        '--profile', type=str, default=None, help="Profiler output"
    )

    args = parser.parse_args()

    model_config = ModelConfig.from_file(
        args.config, args.run_number, continue_training=getattr(args, 'continue'), device=args.device, seed=args.seed,
        params={k: v for (k, v) in (Parser.parse_equality(eq) for eq in args.param)}
    )

    if model_config.project_dir not in sys.path:
        sys.path.append(model_config.project_dir)

    multiprocessing_setting = model_config.provide_with_default('multiprocessing', default=None)

    if multiprocessing_setting:
        # This needs to be called before any of PyTorch module is imported
        multiprocessing.set_start_method(multiprocessing_setting)

    # Set seed already in the launcher
    from vel.util.random import set_seed
    set_seed(model_config.seed)

    model_config.banner(args.command)

    if args.profile:
        print("[PROFILER] Running Vel in profiling mode, output filename={}".format(args.profile))
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        model_config.run_command(args.command, args.varargs)
        profiler.disable()

        profiler.dump_stats(args.profile)
        profiler.print_stats(sort='tottime')

        print("======================================================================")
        pstats.Stats(profiler).strip_dirs().sort_stats('tottime').print_stats(30)
        print("======================================================================")
        pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(30)
    else:
        model_config.run_command(args.command, args.varargs)

    model_config.quit_banner()


if __name__ == '__main__':
    main()
