import numpy as np
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
from graphviz import Digraph


class StateNode:
    def __init__(self, state):
        self.state = state
        self.num_visited = 0
        self.num_won = 0


class TreeNode:
    def __init__(self, state_node: StateNode, depth: int, parent=None):
        self.state_node = state_node
        self.action2tree_node = {}
        self.depth = depth
        self.parent = parent

    def choose_child(self, c: float):
        """ Picks child based on UCB
            :arg c: float - exploration parameter
            :return best child or None
        """
        best_choice = None
        best_ucb = 0
        for child in self.action2tree_node.values():
            expexted_win = child.state_node.num_won / child.state_node.num_visited
            ucb = expexted_win + c * np.sqrt(2 * np.log(self.state_node.num_visited) / child.state_node.num_visited)

            best_choice = child if ucb > best_ucb else best_choice
            best_ucb = max(ucb, best_ucb)

        return best_choice

    def add_child(self, action, state_node):
        """ Adds child.
            :return new child
        """
        self.action2tree_node[action] = TreeNode(state_node, self.depth + 1, self)
        return self.action2tree_node[action]

    @property
    def state(self):
        return self.state_node.state

    def get_graph(self, g):
        map_moves = ["left", "right", "down", "up"]

        num = 1 - (self.state_node.num_won) / (self.state_node.num_visited)
        num = np.sqrt(num) * 255
        color = hex(int(num))[2:]
        color = "0" + color if len(color) == 1 else color
        color = "#" + "ff" + color + color

        g.node(str(self), fillcolor=color, style="filled",
               label=f"{self.state_node.num_won}/{self.state_node.num_visited}")
        for k, v in self.action2tree_node.items():
            g.edge(str(self), str(v), label=map_moves[k])
            with g.subgraph() as s:
                v.get_graph(s)


class SokoMCTS:
    def __init__(self, *,
                 env: SokobanEnvFast,
                 c: float,
                 max_depth: int,
                 number_of_simulations: int):
        self.env = env
        self.c = c
        self.max_depth = max_depth
        self.number_of_simulations = number_of_simulations

        # State can be in multiple tree nodes so we store all explored states
        self.explored_states2state_node = {}
        env.reset()
        init_state = env.clone_full_state()
        self.root_node = TreeNode(self.get_state_node(init_state), 0)   # For now we assume that seed is fixed
        pass

    def get_state_node(self, state) -> StateNode:
        if state not in self.explored_states2state_node:
            self.explored_states2state_node[state] = StateNode(state)

        return self.explored_states2state_node[state]

    def run(self, passes=1, verbose=0):
        """ Run multiple passes of MCTS, standard procedure:
            - choose leaf
            - perform random simulations in it
            - expand one node per simulation
            - backprop results
            - repeat
        """
        for i in range(passes):
            leaf = self._descend()

            for _ in range(self.number_of_simulations):
                action, done, _ = self._simulate(leaf)
                new_node = self._expand(leaf, action)
                self._backprop(new_node, done)
            if verbose:
                print(f"pass {i}")

    def _descend(self) -> TreeNode:
        """ Go from root node to leaf node. Pick children by computing their UCB """
        current_node = self.root_node
        new_best_node = current_node.choose_child(self.c)
        while new_best_node:
            current_node = new_best_node
            new_best_node = current_node.choose_child(self.c)

        return current_node

    def _simulate(self, start_node) -> (int, bool, float):
        """ Run one random simulation until termination or reaching max_depth
            :return (first action, if episode ended, sum of rewards only in this rollout)
        """
        current_depth = start_node.depth
        action_space = self.env.action_space
        self.env.restore_full_state(start_node.state)

        first_action = action_space.sample()
        _, rew, done, _ = self.env.step(first_action)
        current_depth += 1
        cumulated_rew = rew

        # if done:
        #     raise Exception("Reached end of game in one step of simulation."
        #                     "Guess that further actions are not needed xd")

        while not done and current_depth < self.max_depth:
            action = action_space.sample()
            _, rew, done, _ = self.env.step(action)
            current_depth += 1
            cumulated_rew += rew

        return first_action, done, cumulated_rew

    def _expand(self, node, action) -> TreeNode:
        """ Create new node """
        self.env.restore_full_state(node.state)
        self.env.step(action)
        state_node = self.get_state_node(self.env.clone_full_state())
        return node.add_child(action, state_node)

    def _backprop(self, start_node, won):
        """ Updates values in nodes on path from start_node to root
            :arg start_node - start node (leaf)
            :arg won - whether simulation from start_node's parent ended with success
        """

        start_node.state_node.num_visited += 1
        start_node.state_node.num_won += int(won)
        next_node = start_node.parent

        while next_node is not None:
            next_node.state_node.num_visited += 1
            next_node.state_node.num_won += int(won)
            next_node = next_node.parent

    def get_graph(self):
        g = Digraph("graph")
        self.root_node.get_graph(g)
        return g
