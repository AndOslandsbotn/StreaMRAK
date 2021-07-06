from StreaMRAK.StreaMRAKutil.assist_functions import dist, draw_samples, estimate_span_of_data
import numpy as np
import random

#################################
### Cover tree implementation ###
#################################

def initialization_CoverTree(domain, target):
    """
    Select cover-tree with or without randomizer
    """
    initRadius = estimate_span_of_data(domain, target)
    initPoint, _ = draw_samples(domain, target, n_samples=1)
    initPoint = initPoint[0, :]
    initPath = ()

    return initRadius, initPoint, initPath


class CoverTree():
    def __init__(self, center, target, radius, path, useRandomizer, monitorTheScaleCover):
        self.center = center
        self.target = target
        self.radius = radius
        self.counter = 1  # Number of points covered by the tree
        self.too_far = 0  # ?
        self.path = path
        self.children = []
        self.monitorTheScaleCover = monitorTheScaleCover

        self.useRandomizer = useRandomizer

    def randomizer(self, numChildren):
        """ returns list with randomly shuffled indices."""
        idx = np.arange(numChildren)
        random.shuffle(idx)  # randomly shuffle the indices
        return idx

    def no_of_children(self):
        return len(self.children)

    def get_center(self):
        return self.center

    def get_level(self):
        return len(self.path)

    def get_scale(self):
        return self.radius

    def find_path(self, x):
        return

    def insert(self, x, y):
        return

    def collect_centers(self):
        '''Collect all of the centers defined by a tree
        returns a list where each element is a center, followed by the level of the center'''
        C = [(self.center, len(self.path))]
        if len(self.children) > 0:
            for child in self.children:
                C = C + child.collect_centers()
        return C

    def collect_nodes(self):
        '''returns a list of all nodes in the tree'''
        N = [self]  # N is a list where each element is an coverTree instance
        if len(self.children) > 0:
            for child in self.children:
                N = N + child.collect_nodes()
        return N

    def __str__(self):
        return str(self.path) + ': r=%4.2f, no_child=%d, count=%d' % (self.radius, len(self.children), self.counter)

    def _str_to_level(self, max_level):
        s = self.__str__() + '\n'
        if self.get_level() < max_level and len(self.children) > 0:
            for i in range(len(self.children)):
                s += self.children[i]._str_to_level(max_level)
        return s


class MuffledTree(CoverTree):
    def __init__(self, center, target, radius, path, useRandomizer, monitorTheScaleCover):
        super().__init__(center, target, radius, path, useRandomizer, monitorTheScaleCover)
        self.covered_fraction = 0
        self.punch_through = False
        self.alpha = 0.1
        self.thr = 0.70
        self.parent = None

        self.root = None
        self.hardness = 2
        self.num_occupied = 0

        self.useRandomizer = useRandomizer

        self.listOfPTnodesDict = []
        self.monitorTheScaleCover = monitorTheScaleCover

    def threshold(self, x):
        return 1 / (1 + np.exp(-self.hardness * np.tan(np.pi * (x - 1 / 2))))

    def ignore_none(self, dist, radius):
        """This give back the tree which block all nodes that overlap
         with children from neighbouring parents"""
        if (dist <= radius):
            return False
        else:
            return True

    def ignore_ignoreAll(self, dist, radius):
        """This give back the tree which does not check for
        overlap with children from neighbouring parents"""
        if (dist <= radius):
            return True
        else:
            return True

    def ignore(self, dist, radius):
        """
        Returns whether to ignore that a node is too close to the new point x.
        :param dist:
        :param radius:
        :return:
        """
        if (dist <= radius):
            prob_ignore = self.threshold(dist / radius)
            prob_not_ignore = 1 - prob_ignore
            ignore = np.random.choice([True, False], p=[prob_ignore, prob_not_ignore])
        else:
            ignore = True
        return ignore

    def collect_potential_neighbours(self, x, lvl, neighbours):
        """
        Recursively traverse the tree and collects the nodes that are within reach of the new point
        when we have nodes that are at level lvl, we collect nodes into the neighbour list
        :param x: new point
        :param lvl: level of the potential parent of the new point x
        :param neighbours: Neighbours of the potential parent of the new point x
        """
        for child in self.children:
            radius_thr = (child.radius + self.root.radius / 2 ** (lvl + 1))
            distance = dist(x, child.center)
            if self.ignore(distance, radius_thr):
                continue
            else:
                if child.get_level() == lvl:
                    neighbours.append(child)
                else:
                    child.collect_potential_neighbours(x, lvl, neighbours)
        return

    def collect_neighbours(self, x):
        """list of neighbours of the potential parent of the new point x"""
        neighbours = []
        lvl = self.get_level()
        self.root.collect_potential_neighbours(x, lvl, neighbours)
        if self in neighbours:
            neighbours.remove(self)
        return neighbours

    def check_if_occupied(self, x):
        if self.parent == None:  # Root node
            return 'Free'
        else:
            neighbours = self.collect_neighbours(x)
            for nb in neighbours:
                for child in nb.children:
                    radius_thr = child.radius
                    distance = dist(x, child.center)
                    if self.ignore(distance, radius_thr):
                        continue
                    else:
                        return 'Occupied'
            return 'Free'

    def find_path(self, x):
        """Travers the tree, and check which of the existing
        balls the new point x lands within. The find_path recursion stops when
        we reach a node which has len(self.children) == 0, i.e. no new children.
        we then return this node as [self] as the end of the path (i.e. a leaf node)"""
        d = dist(x, self.center)

        if d > self.radius:
            return None
        if len(self.children) == 0:
            self.monitorTheScaleCover.update_cf_at_level(self.get_level(), newNode=False, passingThrough=False)
            return [self]
        else:
            self.monitorTheScaleCover.update_cf_at_level(self.get_level(), newNode=False, passingThrough=True)
            for child in self.children:
                child_path = child.find_path(x)
                if child_path is None:
                    continue
                else:
                    return [self] + child_path
            return [self]

    def find_path_with_RM(self, x):
        """Travers the tree, and check which of the existing
        balls the new point x lands within. The find_path recursion stops when
        we reach a node which has len(self.children) == 0, i.e. no new children.
        we then return this node as [self] as the end of the path (i.e. a leaf node)

        NB! We add a randomization step, to prevent child_1 to always be checked first
        """
        d = dist(x, self.center)
        if d > self.radius:
            return None

        if len(self.children) == 0:
            self.monitorTheScaleCover.update_cf_at_level(self.get_level(), newNode=False, passingThrough=False)
            if self.check_if_occupied(x) == 'Occupied':
                return 'Occupied'
            else:
                return [self]
        else:
            self.monitorTheScaleCover.update_cf_at_level(self.get_level(), newNode=False, passingThrough=True)

            # We add a randomization step, to prevent child_1 to always be checked first
            indices = self.randomizer(self.no_of_children())
            for idx in indices:
                child = self.children[idx]
                child_path = child.find_path_with_RM(x)
                if child_path is None:
                    continue
                elif child_path == 'Occupied':
                    return 'Occupied'
                else:
                    return [self] + child_path

            if self.check_if_occupied(x) == 'Occupied':
                return 'Occupied'
            else:
                return [self]

    def insert(self, x, y):
        """
        Function to handle if x contains more than 1 single point.
        We then iterate over these points and insert 1 at the time
        :param x: Input points, possibly containing n > 1 points.
        :param y: Target corresponding to x
        :param queue: a queue to store nodes
        :return:
        """

        n, _ = x.shape
        for i in range(n):
            self.insertSinglePoint(x[i, :], y[i, :])

    def insertSinglePoint(self, x, y):
        path = self.find_path_with_RM(x)

        if path is 'Occupied':
            self.num_occupied += 1
            if self.num_occupied % 10000 == 0:
                print("Path is Occupied")
                print("Number of occupied occurances: ", self.num_occupied)
            return False

        if path is None:
            print("We are outside of root node")
            print("Distance between root and new sample", dist(x, self.center))
            print("Radius at root: ", self.radius)
            return False

        # Found a non-trivial path
        leaf = path[-1]
        is_root = len(path) == 1
        if is_root:
            new = MuffledTree(x, y, leaf.radius / 2, leaf.path + (leaf.no_of_children(),), self.useRandomizer,
                                        self.monitorTheScaleCover)
            new.parent = leaf
            new.root = leaf.root
            leaf.children.append(new)

            self.monitorTheScaleCover.update_n_nodes_at_lvl(new.get_level())
            self.monitorTheScaleCover.update_suff_lm_cover()
        else:  # not root
            parent = path[-2]
            if parent.punch_through:
                new = MuffledTree(x, y, leaf.radius / 2, leaf.path + (leaf.no_of_children(),),
                                            self.useRandomizer, self.monitorTheScaleCover)
                new.parent = leaf
                new.root = leaf.root
                leaf.children.append(new)

                self.monitorTheScaleCover.update_n_nodes_at_lvl(new.get_level())
                self.monitorTheScaleCover.update_cf_at_level(new.get_level(), newNode=True, passingThrough=False)
                self.monitorTheScaleCover.update_suff_lm_cover()
                if not leaf.punch_through:
                    # If we have not covered enough of leaf, we should update its cover fraction.
                    # Of course we could also update if leaf.punch_through is true, but this will have no
                    # effect, since the punch through latch is already open...
                    leaf.covered_fraction = (1 - self.alpha) * leaf.covered_fraction

            else:  # don't add new node, instead, update parent statistics
                parent.covered_fraction = (1 - self.alpha) * parent.covered_fraction + self.alpha
                if not parent.punch_through and parent.covered_fraction > self.thr:
                    # When parent becomes punched through
                    # print('node' + str(parent.path) + \
                    #      'punched through frac=%7.5f, count= %d, siblings=%2d' % (
                    #      parent.covered_fraction, parent.counter, parent.no_of_children()))
                    parent.punch_through = True  # This is a latch, once the leaf is punched through it remains so forever

                    # Update PT nodes(landmarks) on parent.get_level()
                    self.monitorTheScaleCover.update_n_lm_at_level(parent.get_level())
                    self.monitorTheScaleCover.update_suff_lm_cover()
        for node in path:
            node.counter += 1
        return True

    def __str__(self):
        return str(self.path) + ': r=%4.2f, no_child=%d, count=%d, cov_frac=%4.3f, punch_through=%1d' \
               % (self.radius, len(self.children), self.counter, self.covered_fraction, int(self.punch_through))


class DampedCoverTreeMaster(MuffledTree):
    def __init__(self, center, target, radius, path, useRandomizer, monitorTheScaleCover):
        super().__init__(center, target, radius, path, useRandomizer, monitorTheScaleCover)

    def select_lm(self, level):
        """
        Select all punched through nodes at level. Then select C*sqrt(n) of these nodes.
        Here n is the current number of nodes (both punched through and not) at the level.
        :param level: The level in the coverTree from which we choose punch through nodes as landmarks
        :return: landmarks (punched through nodes at level)
        """
        Nodes = self.collect_nodes()
        pot_lm = []
        nodes_at_lvl = []
        scale = 1
        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == level:
                nodes_at_lvl.append(node)
                scale = node.get_scale()
                if node.punch_through:
                    pot_lm.append(node.center)
        num_nodes = len(nodes_at_lvl)
        num_pot_lm = len(pot_lm)
        print("num_pot_lm in new algo: ", num_pot_lm)
        pot_idx = np.arange(num_pot_lm)
        num_lm = min(int((self.monitorTheScaleCover.lm_to_node_ratio) * np.sqrt(num_nodes)), num_pot_lm)
        idx = np.random.choice(pot_idx, replace=False, size=num_lm)
        lm = [pot_lm[i] for i in idx]
        print("Num landmarks selected in new algo: ", len(lm))
        lm = np.array(lm)
        pot_lm = np.array(pot_lm)
        return lm, pot_lm, scale

    def select_lm_old_and_new(self, level):
        """
        Select all nodes at level, which are punched through, as landmarks
        :param level: The level in the coverTree from which we choose punch through nodes as landmarks
        :return: landmarks (punched through nodes at level)
        """
        Nodes = self.collect_nodes()
        lmList = []

        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == level:
                scale = node.get_scale()
                if node.punch_through:
                    lmList.append(node.center)

        _, _, _ = self.select_lm_new(level)

        return np.array(lmList), np.array(lmList), scale

    def select_nodes(self, level):
        Nodes = self.collect_nodes()
        nodes_at_lvl = []
        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == level:
                nodes_at_lvl.append(node.center)
        return np.array(nodes_at_lvl)

    def select_all_nodes(self):
        '''returns all centers and corresponding targets in tree'''
        Nodes = self.collect_nodes()
        n_nodes = len(Nodes)
        centers_at_lvl = [None] * n_nodes
        targets_at_lvl = [None] * n_nodes
        t = 0
        for node in Nodes:
            centers_at_lvl[t] = node.center
            targets_at_lvl[t] = node.target
            t += 1
        return np.array(centers_at_lvl), np.array(targets_at_lvl)

    def select_nodes_as_trData(self, lvl):
        Nodes = self.collect_nodes()
        centers_at_lvl = []
        targets_at_lvl = []

        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == lvl:
                centers_at_lvl.append(node.center)
                targets_at_lvl.append(node.target)
        return np.array(centers_at_lvl), np.array(targets_at_lvl)

    def get_numNodesAtLvl(self, level):
        numNodesAtLvl = 0
        Nodes = self.collect_nodes()
        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == level:
                numNodesAtLvl += 1
        return numNodesAtLvl

    def get_numLmAtLvl(self, level):
        numLmAtLvl = 0
        Nodes = self.collect_nodes()
        for node in Nodes:
            nodeLevel = len(node.path)
            if nodeLevel == level:
                if node.punch_through:
                    numLmAtLvl += 1
        return numLmAtLvl

