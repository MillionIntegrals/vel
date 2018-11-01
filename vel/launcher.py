#!/usr/bin/env python
import argparse
import datetime as dtm

from vel.api import ModelConfig
from vel.util.random import set_seed
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

    args = parser.parse_args()

    model_config = ModelConfig.from_file(
        args.config, args.run_number, continue_training=getattr(args, 'continue'), device=args.device, seed=args.seed,
        params={k: v for (k, v) in (Parser.parse_equality(eq) for eq in args.param)}
    )

    print(model_config)

    # Set seed already in the launcher
    set_seed(model_config.seed)

    model_config.banner(args.command)
    model_config.run_command(args.command, args.varargs)
    model_config.quit_banner()


if __name__ == '__main__':
    main()
