"""
search.py
---------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).

Modified and extended by Amin D. Alamdari (amin.alamdari@ozu.edu.tr), 2026.
Changes: Project restructuring, modernized Python packaging, and updated
assignment scaffolding. See README.md for full list of changes.
"""

"""
In search.py, you will implement generic search algorithms which are called by Pacman agents (in search_agents.py).
"""

import pacman.util as util
from pacman.engine import GameState
from pacman.game import Actions


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        GameState.get_pacman_position

        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, step_cost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'step_cost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tiny_maze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tiny_maze.
    """
    from pacman.game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depth_first_search(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """

    visited = set()
    stack = util.Stack()
    start_node = problem.get_start_state()

    if problem.is_goal_state(start_node):
        return []
    
    stack.push((start_node,[]))

    while stack.is_empty() != True:
        state, path = stack.pop()

        if state in visited:
            continue
        
        visited.add(state)

        if problem.is_goal_state(state):
            return path  # we have reached the goal node
        
        for successor, action, step_cost in problem.get_successors(state):
            if successor not in visited:
                stack.push((successor, path + [action]))
    #util.raise_not_defined()
    return []



def breadth_first_search(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # TODO: Implement breadth-first graph search using util.Queue

    visited = set()
    queue = util.Queue()
    start_state = problem.get_start_state()

    if problem.is_goal_state(start_state):
        return []
    queue.push((start_state,[]))

    while not queue.is_empty():
        state, path = queue.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.is_goal_state(state):
            return path
        
        for successor, action, cost in problem.get_successors(state):
            if successor not in visited:
                queue.push((successor, path + [action]))

    #util.raise_not_defined()
    return []


def uniform_cost_search(problem: SearchProblem):
    """Search the node of least total cost first."""
    # TODO: Implement uniform-cost graph search using util.PriorityQueue
    p_Queue = util.PriorityQueue()
    start_state = problem.get_start_state()
    visited_cost = {start_state : 0}
    
    if problem.is_goal_state(start_state):
        return []
    
    p_Queue.push((start_state,[],0), 0)

    while not p_Queue.is_empty():
        state, path, path_cost = p_Queue.pop()

        if state in visited_cost and visited_cost[state] < path_cost:
            continue

        if problem.is_goal_state(state):
            return path

        """lowest_item = min(p_Queue, key = lambda x: x[1])
        item_to_expand = lowest_item[0]"""

        for successor, action, cost in problem.get_successors(state):
            new_cost = path_cost + cost
            if successor not in visited_cost or visited_cost[successor] > new_cost:
                visited_cost[successor] = new_cost
                p_Queue.push((successor, path + [action], new_cost),new_cost)

    #util.raise_not_defined()
    return []


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem: SearchProblem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # TODO: Implement A* graph search using util.PriorityQueue with f(n) = g(n) + h(n)

    p_Queue = util.PriorityQueue()
    start_state = problem.get_start_state()
    visited_cost = {start_state: 0}

    if problem.is_goal_state(start_state):
        return []
    
    p_Queue.push((start_state, [], 0, 0),0)
    
    while not p_Queue.is_empty():
        state, path, path_cost, f_x = p_Queue.pop()

        if state in visited_cost and visited_cost[state] < f_x:
            continue

        if problem.is_goal_state(state):
            return path
        

        for successor, action, cost in problem.get_successors(state):
            h_x = heuristic(successor,problem)
            g_x = path_cost + cost
            f_x = g_x + h_x
            if successor not in visited_cost or visited_cost[successor] > f_x:
                visited_cost[successor] = f_x
                p_Queue.push((successor, path + [action], g_x, f_x),f_x)

    #util.raise_not_defined()
    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
