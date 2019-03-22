#!/usr/bin/env python

from archngv import NGVConfig


def main(config_path):
    """ Given a config file generate the directories and copy the config there
    """
    config = NGVConfig.from_file(config_path)

    config.create_directories()
    config.save()
    config.generate_circuit_config()


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
