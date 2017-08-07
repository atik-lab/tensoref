"""
Tensoref

Usage:
    tensoref fire-theft (--linear|--polynomial) [options]
    tensoref style-transfer <content> <style> <width> <height>
    tensoref --help

Options:
    -v, --verbose                   Verbose mode
    -d, --data-dir <level>          Set data directory
    -l, --logging-level <level>     Set logging level (debug, info, warning or error)
    -h, --help                      Show this screen
    --version                       Show version
"""
import logging
import os
from docopt import docopt

from tensoref.config import CONFIG
from tensoref import __version__

import tensoref.problems.fire_theft as fire_theft
import tensoref.problems.style_transfer as style_transfer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # avoid tensorflow warnings


def main():
    """
    Main entry point
    """
    args = docopt(__doc__, version=__version__)  # docopt documentation https://pypi.python.org/pypi/docopt/

    # set argument options
    if args['--verbose']:
        CONFIG.set('general', 'verbose', 'true')
    if args['--data-dir']:
        CONFIG.set('general', 'data-dir', args['--data-dir'])
    if args['--logging-level']:
        if args['--logging-level'] == 'debug':
            CONFIG.set('logging', 'level', '10')
        if args['--logging-level'] == 'info':
            CONFIG.set('logging', 'level', '20')
        if args['--logging-level'] == 'warning':
            CONFIG.set('logging', 'level', '30')
        if args['--logging-level'] == 'error':
            CONFIG.set('logging', 'level', '40')

    # start logging
    _start_logging()
    logging.debug("Tensoref started.")

    # execute command
    if args['fire-theft']:
        if args['--linear']:
            fire_theft.run_linear()
        elif args['--polynomial']:
            fire_theft.run_polynomial()
        else:
            print('Please specify --linear or --polynomial')
    elif args['style-transfer']:
        style_transfer.run(args['<content>'], args['<style>'], args['<width>'], args['<height>'])
    else:
        # --- default ---
        print('Please specify a valid command.')
        print('For more info execute: tensoref --help')

    logging.debug("Tensoref finished successfully")


def _start_logging():
    """
    Load the logging with the config if enabled in the configuration file
    https://docs.python.org/3.5/library/logging.html
    """
    if CONFIG.getboolean('logging', 'enabled'):
        # set log file logger
        logging.basicConfig(
            filename=os.path.dirname(os.path.realpath(__file__)) + '/' + CONFIG.get('logging', 'file'),
            level=CONFIG.getint('logging', 'level'),
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        # add output handler
        root_logger = logging.getLogger()
        console_handler = logging.StreamHandler()
        if CONFIG.getboolean('general', 'verbose'):
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
    else:
        print('Logging disabled...')

# start of the application
if __name__ == "__main__":
    main()
