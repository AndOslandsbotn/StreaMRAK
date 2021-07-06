import csv
import os
from os import path
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

class MRFALKONconfig():
    def __init__(self, verbosity):
        self.tag = '_default'

        # Set verbosity
        self.verbosity = verbosity

        # data loggers config
        fileNameLogger = 'loggerConfig'
        self.loggerConfig = loadConfigFromCSV(fileNameLogger, self.tag, verbosity)

        # FALKON solver config
        fileNameFalkon = 'falkonConfig'
        self.falkon_conf = loadConfigFromCSV(fileNameFalkon, self.tag, verbosity)

class StreaMRAKconfig():
    def __init__(self, verbosity):
        self.tag = '_default'

        # Set verbosity
        self.verbosity = verbosity

        # data loggers config
        fileNameLogger = 'loggerConfig'
        self.loggerConfig = loadConfigFromCSV(fileNameLogger, self.tag, verbosity)

        # FALKON solver config
        fileNameFalkon = 'falkonConfig'
        self.falkon_conf = loadConfigFromCSV(fileNameFalkon, self.tag, verbosity)

        # monitorScaleCover config
        fileNameMonitorScale = 'monitorScaleCovConfig'
        self.monitor_conf = loadConfigFromCSV(fileNameMonitorScale, self.tag, verbosity)

        # CoverTree object
        fileNameCoverTree = 'coverTreeConfig'
        self.coverTree_conf = loadConfigFromCSV(fileNameCoverTree, self.tag, verbosity)


def loadConfigFromCSV(fileName, fileNameTag, verbosity):
    """
    Reads text file formated as csv with: key, value

    :param path: Path of the text file
    :param textFile: Name of the text file
    :return: configDict
     """

    directoryConfig = dir_path

    fileName = fileName + fileNameTag
    dir_and_fileName = path.join(directoryConfig, fileName)

    reader = csv.reader(open(dir_and_fileName))
    configDict = {}
    for row in reader:
        if len(row) != 0:
            if row[0][0] != '#':
                key = row[0]
                if key in configDict:  # In case duplicate
                    pass
                try:
                    configDict[key] = float(row[1])
                except:
                    configDict[key] = row[1].strip()
    if verbosity > 0:
        print_config(configDict)
    return configDict


def print_config(config):
    print(f'{config["tag"]} configurations: ')
    for key in config:
        print('{:>30}: {:>30}'.format(key, str(config[key])))
    print('\n')
    return