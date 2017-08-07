"""
Parse config file with configparser
https://docs.python.org/3/library/configparser.html
"""
import os
import sys
from configparser import ConfigParser

DEFAULT_CONFIG_FILE = 'tensoref.ini'


def create_config():
    """
    Parse default configuration file
    """
    parser = ConfigParser()
    config_filepath = os.path.dirname(os.path.realpath(__file__)) + '/' + DEFAULT_CONFIG_FILE
    read = parser.read(config_filepath)
    if not read:
        print('Configuration file ' + DEFAULT_CONFIG_FILE + ' not found. Quiting...')
        sys.exit(0)
    return parser

# Create config variable accessible with from "tensoref.config import CONFIG"
CONFIG = create_config()
