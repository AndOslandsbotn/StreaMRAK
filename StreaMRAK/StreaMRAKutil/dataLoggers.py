import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt
from decimal import Decimal
import itertools as iter
from pathlib import Path

#System
import os
from os import path
import io

#Database
import sqlite3
from sqlite3 import Error
from sqlalchemy import create_engine

class DatabasePool():
    """
    Creates a database pool for storing handling connections to a
    SQLite database.
    """
    _nInstances = iter.count(start=0, step=1)
    def __init__(self, ExpName):
        self.id = next(self._nInstances)
        self.expName = ExpName #  Name of experiment

        self.cwd = os.getcwd()
        self.dataDir = path.join(self.cwd, 'Databases')
        try:
            os.stat(self.dataDir)
        except:
            os.mkdir(self.dataDir)

        self.dbPoolName = f'dbPool_{self.expName}_{self.id}'
        self.file = path.join(self.dataDir, self.dbPoolName)
        try:
            if os.path.isfile(self.file):
                raise ValueError
        except ValueError as err:
            raise ValueError(f'Experiment {ExpName} is already running, try a different experiment name')

        print("Database pool: ", self.dbPoolName)

        self.create_PoolEngine()
        return

    def get_poolEngine(self):
        return self.poolEngine

    def create_PoolEngine(self):
        "Establish a database pool"
        try:  # If database pool with this name already exists, we remove it
            os.remove(self.file)
        except OSError:
            pass

        db = 'sqlite:///' + self.file
        if os.path.isfile(db):
            raise Exception(f'The database {db} already exists. Try running the code'
                            f'using a different database name')
        try:
            self.poolEngine = create_engine(db)
        except Error as e:
            print(f'Could not establish pool for database {self.file}, error {e}')

    def create_connection_to_pool(self):
        try:
            connection = self.poolEngine.raw_connection()
        except Error as e:
            print(f'Could not connect to {self.file}, error {e}')
        finally:
            cursor = connection.cursor()
        return cursor, connection

    def dispose_pool(self):
        """Close the database pool"""
        self.poolEngine.dispose()
        return


############################################
### Utilities for the datalogger classes ###
############################################
class LoggerUtilities():
    """
    Utilities for the logger classes
    """
    def __init__(self):
        return

    def decode_numpy_from_bytes(self, lm_bytes):
        return np.fromstring(lm_bytes, dtype=float)

    def encode_numpy_to_bytes(self, array):
        return array.tostring()


##########################################
### Data logger for the landmarks algo ###
##########################################
class DataLoggerLM(LoggerUtilities):
    def __init__(self, config, databasePool):
        super().__init__()

        self.tableName = config['landmarksTable'].strip()
        self.databasePool = databasePool

        # Initialize a table
        cursor, connection = self.databasePool.create_connection_to_pool()
        self.create_SQLite_table(cursor)
        connection.close()
        return

    def create_SQLite_table(self, cursor):
        sql_quiry = f'CREATE TABLE IF NOT EXISTS {self.tableName} (level INT, scale FLOAT, landmarks BLOB)'
        cursor.execute(sql_quiry)
        return

    def store_lm_and_scale_at_level(self, level, scale, landmarks):
        """
        Stores the landmarks at scale
        :param scale: scale in question
        :param landmarks: landmarks to store in database
        :return:
        """
        cursor, connection = self.databasePool.create_connection_to_pool()
        sql_quiry = f'INSERT INTO {self.tableName} VALUES (?, ?, ?)'

        n, d = landmarks.shape
        for i in range(0, n):
            lm_bytes = self.encode_numpy_to_bytes(landmarks[i])
            cursor.execute(sql_quiry, (level, scale, lm_bytes))
        connection.commit()
        connection.close()
        return

    def select_lm_and_scale_at_level(self, level):
        cursor, connection = self.databasePool.create_connection_to_pool()
        sql_quiry = f'SELECT * FROM {self.tableName} WHERE level = {level}'
        cursor.execute(sql_quiry)

        landmarkList = []
        for tup in cursor.fetchall():
            level, scale, lm_bytes = tup
            landmark = self.decode_numpy_from_bytes(lm_bytes)
            landmarkList.append(landmark)
        landmarks = np.array(landmarkList)

        connection.close()
        return landmarks, scale


#######################################
### Data logger for the model coef  ###
#######################################
class DataLoggerCoef(LoggerUtilities):
    def __init__(self, config, databasePool):
        super().__init__()

        self.tableName = config['coefTable'].strip()
        self.databasePool = databasePool

        # Initialize a table
        cursor, connection = self.databasePool.create_connection_to_pool()
        self.create_SQLite_table(cursor)
        connection.close()
        return

    def create_SQLite_table(self, cursor):
        sql_quiry = f'CREATE TABLE IF NOT EXISTS {self.tableName} (level INT, scale FLOAT, coef BLOB)'
        cursor.execute(sql_quiry)
        return

    def store_coef_at_level(self, level, scale, coef):
        """
        Stores the landmarks at scale
        :param scale: scale in question
        :param landmarks: landmarks to store in database
        :return:
        """
        cursor, connection = self.databasePool.create_connection_to_pool()
        sql_quiry = f'INSERT INTO {self.tableName} VALUES (?, ?, ?)'

        n, d = coef.shape
        for i in range(0, n):
            coef_bytes = self.encode_numpy_to_bytes(coef[i])
            cursor.execute(sql_quiry, (level, scale, coef_bytes))
        connection.commit()
        connection.close()
        return

    def select_coef_at_level(self, level):
        cursor, connection = self.databasePool.create_connection_to_pool()
        sql_quiry = f'SELECT * FROM {self.tableName} WHERE level = {level}'
        cursor.execute(sql_quiry)

        coefList = []
        for tup in cursor.fetchall():
            level, scale, coef_bytes = tup
            coef = self.decode_numpy_from_bytes(coef_bytes)
            coefList.append(coef)
        coef = np.array(coefList)

        connection.close()
        return coef



#######################################
### Data logger for the predictions ###
#######################################
class DataLoggerProgr():
    def __init__(self, ExpName):
        self.cwd = os.getcwd()
        self.directory = os.path.join(self.cwd, 'LoggModel')
        try:
            os.stat(self.directory)
        except:
            os.mkdir(self.directory)
        return

    def logg_lm(self, lm, lvl, ExpName):
        path = os.path.join(self.directory, ExpName)
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        path = os.path.join(path, f'lm_lvl{lvl}')
        np.savetxt(path, lm, delimiter=",")
        return

    def logg_lm_target(self, lm_target, lvl, ExpName):
        path = os.path.join(self.directory, ExpName)
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        path = os.path.join(path, f'lm_target_lvl{lvl}')
        np.savetxt(path, lm_target, delimiter=",")
        return

    def logg_coef(self, coef, lvl, ExpName):
        path = os.path.join(self.directory, ExpName)
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        path = os.path.join(path, f'coef_lvl{lvl}')
        np.savetxt(path, coef, delimiter=",")
        return


class DataLoggerMonitorScaleCover():
    """ This class will be embedded in the coverTree to monitor
    scale cover and the filling of levels in the tree
    """
    def __init__(self, config):
        self.allocSize = 40
        self.levels = np.arange(self.allocSize, dtype=int)
        self.lm_to_node_ratio = float(config['lmNodeRatioFactor'])
        self.intervalStep = int(config['save_at_intervals'])
        self.highRate = float(config['highRate_PTlevel'])
        self.lowRate = float(config['lowRate_PTlevel'])
        self.thr_PT_of_lvl = float(config['thr_PTlevel'])

        self.numPointsCollectedCounter = 0
        self.numPoints_when_suffNodeCover = np.zeros(self.allocSize)
        self.numPoints_when_suffLmCover = np.zeros(self.allocSize)

        self.suff_node_cover = np.zeros(self.allocSize)
        self.suff_node_cover_waiting = []
        self.suff_node_cover_finished = []
        self.suff_lm_cover = np.zeros(self.allocSize)
        self.suff_lm_cover_waiting = []


        self.n_lm_at_lvl = np.zeros(self.allocSize)
        self.n_nodes_at_lvl = np.zeros(self.allocSize)

        self.cf_at_lvl = np.zeros(self.allocSize)

        self.coverTreeSummary = {}
        self.coverTreeSummary['lvl'] = np.zeros(self.allocSize)
        self.coverTreeSummary['NumNodesWhenPT'] = np.zeros(self.allocSize)
        self.coverTreeSummary['NumNodesWhenLMcover'] = np.zeros(self.allocSize)
        self.coverTreeSummary['NumNodesWhenFinish'] = np.zeros(self.allocSize)
        self.coverTreeSummary['NumLMwhenLMcover'] = np.zeros(self.allocSize)
        self.coverTreeSummary['NumLMwhenFinish'] = np.zeros(self.allocSize)

        self.lvl_filling_log = [{}] * self.allocSize
        return

    def log_at_intervals(self):
        """Function that will be called in the coverTree
            to log the filling of the coverTree levels
        """
        self.numPointsCollectedCounter += 1
        if self.numPointsCollectedCounter % 10 == 0 and self.numPointsCollectedCounter < 10000:
            # We enforce special treatment of the first three layers
            self.log_at_current_step(self.numPointsCollectedCounter, self.levels[0:3])

        if self.numPointsCollectedCounter % self.intervalStep == 0:
            self.log_at_current_step(self.numPointsCollectedCounter, self.levels[3:-1])

    def log_at_current_step(self, step, levels):
        """Function that is called by log_at_intervals,
        see above
        """

        for level in levels:
            if level >= self.allocSize:
                self.update_alloc_size()
            if bool(self.lvl_filling_log[level]) == False:
                self.lvl_filling_log[level] = {f'Step{step}': [step, self.n_lm_at_lvl[level],
                                                               self.n_nodes_at_lvl[level],
                                                               self.cf_at_lvl[level]
                                                               ]}
            else:
                self.lvl_filling_log[level][f'Step{step}'] = [step, self.n_lm_at_lvl[level],
                                                              self.n_nodes_at_lvl[level],
                                                              self.cf_at_lvl[level]
                                                              ]

    def save_lvlfilling_log(self, location):
        """Save the logging related to filling of the coverTree levels
        as recorded by the log_at_intervals function, see above
        """
        directory = os.path.join(os.getcwd(), 'LvlFillingLogg', location)
        Path(directory).mkdir(parents=True, exist_ok=True)

        colNames = ['Num collected points', 'Num Landmarks', 'Num Nodes', 'Estimated CF']
        for level in self.levels:
            logsDF = pd.DataFrame.from_dict(self.lvl_filling_log[level], orient='index', columns=colNames)

            fileName = f'Level{level}'
            file = os.path.join(directory, fileName)
            logsDF.to_csv(file)

        fileName = 'numPoints_when_suffNodeCover'
        file = os.path.join(directory, fileName)
        np.savetxt(file, self.numPoints_when_suffNodeCover, delimiter=',')

        fileName = 'numPoints_when_suffLmCover'
        file = os.path.join(directory, fileName)
        np.savetxt(file, self.numPoints_when_suffLmCover, delimiter=',')

    def update_n_nodes_at_lvl(self, lvl):
        if self.allocSize <= lvl:
            self.update_alloc_size()  # If we reach a level that is deeper than the allocated size
        self.n_nodes_at_lvl[lvl] += 1
        return

    def update_n_lm_at_level(self, lvl):
        if self.allocSize <= lvl:
            self.update_alloc_size()  # If we reach a level that is deeper than the allocated size
        self.n_lm_at_lvl[lvl] += 1
        return

    def update_cf_at_level(self, lvl, newNode, passingThrough):
        #print(f'Num nodes: {self.numNodes_at_level}')
        #print(f'Num PT nodes: {self.numPTnodes_at_level}')
        if lvl >= self.allocSize:
            self.update_alloc_size()  # If we reach a level that is deeper than the allocated size

        if self.suff_node_cover[lvl] == False:
            if (newNode == True) and (passingThrough == False):
                T = 0
            elif (newNode == False) and (passingThrough == True):
                #If we are passing through level without generating a new node, then we know
                #that this point was covered at the given level.
                T = 1
            elif (newNode == False) and (passingThrough == False):
                #If the point enter the leaf node,
                # then we know that the point was covered at the level of the leaf.
                # Namely covered by the leaf...
                T = 1
            else:
                print("This combination of newNode and newPTnode don't exist")
            if lvl < 3:
                self.cf_at_lvl[lvl] = (1 - self.highRate) * self.cf_at_lvl[lvl] + T * self.highRate
            else:
                self.cf_at_lvl[lvl] = (1 - self.lowRate) * self.cf_at_lvl[lvl] + T*self.lowRate
            if self.cf_at_lvl[lvl] > self.thr_PT_of_lvl:
                # If the cover fraction of a given level
                # reaches the threshold, then we set the
                # level to Punch Through
                print(f'Level {lvl} punched through with {self.n_nodes_at_lvl[lvl]} nodes')

                for l in range(min(np.where(self.suff_node_cover == False)[0]), lvl + 1):
                    print(f"Include lvl {l} as punched through with {self.n_nodes_at_lvl[l]} nodes")
                    self.set_suff_node_cover(l)

                if lvl >= self.allocSize:
                    self.update_alloc_size()
                self.coverTreeSummary[f'lvl'][lvl] = lvl
                self.coverTreeSummary[f'NumNodesWhenPT'][lvl] = self.n_nodes_at_lvl[lvl]
        return

    def set_suff_lm_cover(self, lvl):
        if lvl >= self.allocSize:
            self.update_alloc_size()
        self.suff_lm_cover[lvl] = True
        self.suff_lm_cover_waiting.append(lvl)

        self.numPoints_when_suffLmCover[lvl] = self.numPointsCollectedCounter
        return

    def set_suff_node_cover(self, lvl):
        if lvl >= self.allocSize:
            self.update_alloc_size()
        self.suff_node_cover[lvl] = True
        self.suff_node_cover_waiting.append(lvl)

        self.numPoints_when_suffNodeCover[lvl] = self.numPointsCollectedCounter
        return

    def update_suff_lm_cover_old(self):
        """
        This function checks if level is sufficiently covered. It does this by first checking that
        the level is punched though, i.e. has enough nodes to sufficiently cover the input domain. It then
        checks if the number of punched though nodes is equal or larger to sqrt(num nodes at level). If this
        is the case then the level is considered sufficiently covered by landmarks (i.e. punched through nodes)
        :param lvl: Level in question
        :return: Whether level is sufficiently covered or not
        """

        if len(self.suff_node_cover_waiting) > 0:
            for lvl in self.suff_node_cover_waiting:
                cond1 = self.n_lm_at_lvl[lvl] >= self.lm_to_node_ratio * np.sqrt(self.n_nodes_at_lvl[lvl])
                cond2 = self.n_lm_at_lvl[lvl] > 0.4 * self.n_nodes_at_lvl[lvl]
                if cond1 or cond2:
                    print(f'Level {lvl} is sufficiently covered with landmarks. '
                        f'With {self.n_nodes_at_lvl[lvl]} nodes and {self.n_lm_at_lvl[lvl]} landmarks')

                    for l in range(min(np.where(self.suff_lm_cover==False)[0]), lvl + 1):
                        print(f"Include lvl {l} as suff covered with {self.n_nodes_at_lvl[l]} nodes and {self.n_lm_at_lvl[l]} landmarks")
                        self.set_suff_lm_cover(l)

                        if l in self.suff_node_cover_waiting:
                            self.suff_node_cover_waiting.remove(l)

                    if lvl >= self.allocSize:
                        self.update_alloc_size()
                    self.coverTreeSummary['NumNodesWhenLMcover'][lvl] = self.n_nodes_at_lvl[lvl]
                    self.coverTreeSummary['NumLMwhenLMcover'][lvl] = self.n_lm_at_lvl[lvl]
                else:
                    pass

    def update_suff_lm_cover(self):
        """
        This function checks if level is sufficiently covered. It does this by first checking that
        the level is punched though, i.e. has enough nodes to sufficiently cover the input domain. It then
        checks if the number of punched though nodes is equal or larger to sqrt(num nodes at level). If this
        is the case then the level is considered sufficiently covered by landmarks (i.e. punched through nodes)
        :param lvl: Level in question
        :return: Whether level is sufficiently covered or not
        """

        if len(self.suff_node_cover_waiting) > 0:
            lvl = self.suff_node_cover_waiting[0]
            cond1 = self.n_lm_at_lvl[lvl] >= self.lm_to_node_ratio * np.sqrt(self.n_nodes_at_lvl[lvl])
            cond2 = self.n_lm_at_lvl[lvl] > 0.4 * self.n_nodes_at_lvl[lvl]

            cond3 = False
            if lvl+1 in self.suff_node_cover_waiting:
                if 2*self.n_nodes_at_lvl[lvl] > self.n_nodes_at_lvl[lvl+1]:
                    print(f"Level {lvl} has 2 times more nodes than level {lvl+1}")
                    cond3 = True
                else:
                    cond3 = False

            if cond1 or cond2 or cond3:
                print(f'Level {lvl} is sufficiently covered with landmarks. '
                        f'With {self.n_nodes_at_lvl[lvl]} nodes and {self.n_lm_at_lvl[lvl]} landmarks')

                for l in range(min(np.where(self.suff_lm_cover==False)[0]), lvl + 1):
                    print(f"Include lvl {l} as suff covered with {self.n_nodes_at_lvl[l]} nodes and {self.n_lm_at_lvl[l]} landmarks")
                    self.set_suff_lm_cover(l)

                    if l in self.suff_node_cover_waiting:
                        self.suff_node_cover_waiting.remove(l)

                if lvl >= self.allocSize:
                    self.update_alloc_size()
                self.coverTreeSummary['NumNodesWhenLMcover'][lvl] = self.n_nodes_at_lvl[lvl]
                self.coverTreeSummary['NumLMwhenLMcover'][lvl] = self.n_lm_at_lvl[lvl]
            else:
                pass

    def update_alloc_size(self):
        self.allocSize = self.allocSize * 2

        temp_n_lm_at_lvl = np.zeros(self.allocSize)
        temp_n_lm_at_lvl[np.arange(0, len(self.n_lm_at_lvl))] = self.n_lm_at_lvl
        self.n_lm_at_lvl = temp_n_lm_at_lvl

        temp_n_nodes_at_lvl = np.zeros(self.allocSize)
        temp_n_nodes_at_lvl[np.arange(0, len(self.n_nodes_at_lvl))] = self.n_nodes_at_lvl
        self.n_nodes_at_lvl = temp_n_nodes_at_lvl

        temp_suff_node_cover = np.zeros(self.allocSize)
        temp_suff_node_cover[np.arange(0, len(self.suff_node_cover))] = self.suff_node_cover
        self.suff_node_cover = temp_suff_node_cover

        temp_suff_lm_cover = np.zeros(self.allocSize)
        temp_suff_lm_cover[np.arange(0, len(self.suff_lm_cover))] = self.suff_lm_cover
        self.suff_lm_cover = temp_suff_lm_cover

        temp_cf_at_lvl = np.zeros(self.allocSize)
        temp_cf_at_lvl[np.arange(0, len(self.cf_at_lvl))] = self.cf_at_lvl
        self.cf_at_lvl = temp_cf_at_lvl

        temp_numPoints_whenLevelIsPT = np.zeros(self.allocSize)
        temp_numPoints_whenLevelIsPT[np.arange(0, len(self.coverFrac_at_level))] = self.numPoints_when_suffNodeCover

        temp_numPoints_whenLevelIsCovered = np.zeros(self.allocSize)
        temp_numPoints_whenLevelIsCovered[np.arange(0, len(self.coverFrac_at_level))] = self.numPoints_when_suffLmCover

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['lvl']))] = self.coverTreeSummary['lvl']
        self.coverTreeSummary['lvl'] = temp

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['NumNodesWhenPT']))] = self.coverTreeSummary['NumNodesWhenPT']
        self.coverTreeSummary['NumNodesWhenPT'] = temp

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['NumNodesWhenLMcover']))] = self.coverTreeSummary['NumNodesWhenLMcover']
        self.coverTreeSummary['NumNodesWhenLMcover'] = temp

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['NumNodesWhenFinish']))] = self.coverTreeSummary['NumNodesWhenFinish']
        self.coverTreeSummary['NumNodesWhenFinish'] = temp

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['NumLMwhenLMcover']))] = self.coverTreeSummary['NumLMwhenLMcover']
        self.coverTreeSummary['NumLMwhenLMcover'] = temp

        temp = np.zeros(self.allocSize)
        temp[np.arange(0, len(self.coverTreeSummary['NumLMwhenFinish']))] = self.coverTreeSummary['NumLMwhenFinish']
        self.coverTreeSummary['NumLMwhenFinish'] = temp





