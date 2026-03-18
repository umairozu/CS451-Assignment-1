"""
search_agents.py
---------------
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
from __future__ import annotations

"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depth_first_search

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"TODO"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import Any, Callable, Optional
from pacman.game import Directions
from pacman.game import Agent
from pacman.game import Actions
from pacman.game import Grid
import pacman.util as util
import time
import pacman.search as search
import pacman
import itertools
import statistics


class GoWestAgent(Agent):
    """An agent that goes West until it can't."""

    def __init__(self) -> None:
        super().__init__()

    def get_action(self, state: pacman.GameState) -> str:
        """The agent receives a GameState (defined in pacman.py)."""
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depth_first_search or dfs
      breadth_first_search or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn: str = 'depth_first_search', prob: str = 'PositionSearchProblem',
                 heuristic: str = 'null_heuristic') -> None:
        super().__init__()
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.search_function: Optional[Callable] = func
        else:
            if heuristic in globals():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in search_agents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.search_function = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in search_agents.py.')
        self.search_type: type = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)
        self.actions: list[str] = []
        self.action_index: int = 0

    def register_initial_state(self, state: pacman.GameState) -> None:
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.search_function is None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.search_type(state)  # Makes a new search problem
        self.actions = self.search_function(problem)  # Find a path
        if self.actions is None:
            self.actions = []
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (total_cost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % getattr(problem, '_expanded', 0))

    def get_action(self, _state: pacman.GameState) -> str:
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, game_state: pacman.GameState, cost_fn: Callable[[tuple[int, int]], float] = lambda x: 1,
                 goal: tuple[int, int] = (1, 1), start: Optional[tuple[int, int]] = None, warn: bool = True,
                 visualize: bool = True) -> None:
        """
        Stores the start and goal.

        game_state: A GameState object (pacman.py)
        cost_fn: A function from a search state (tuple) to a non-negative number
        goal: A position in the game_state
        """
        self.walls: Grid = game_state.get_walls()
        self.start_state: tuple[int, int] = game_state.get_pacman_position()
        if start is not None: self.start_state = start
        self.goal: tuple[int, int] = goal
        self.cost_fn: Callable[[tuple[int, int]], float] = cost_fn
        self.visualize: bool = visualize
        if warn and (game_state.get_num_food() != 1 or not game_state.has_food(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited: dict[tuple[int, int], bool] = {}
        self._visited_list: list[tuple[int, int]] = []
        self._expanded: int = 0  # DO NOT CHANGE

    def get_start_state(self) -> tuple[int, int]:
        return self.start_state

    def is_goal_state(self, state: tuple[int, int]) -> bool:
        is_goal = state == self.goal

        # For display purposes only
        if is_goal and self.visualize:
            self._visited_list.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'draw_expanded_cells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.draw_expanded_cells(self._visited_list)  # @UndefinedVariable

        return is_goal

    def get_successors(self, state: tuple[int, int]) -> list[tuple[tuple[int, int], str, float]]:
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, step_cost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'step_cost' is the incremental
         cost of expanding to that successor
        """

        successors: list[tuple[tuple[int, int], str, float]] = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                cost = self.cost_fn(next_state)
                successors.append((next_state, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visited_list.append(state)

        return successors

    def get_cost_of_actions(self, actions: Optional[list[str]]) -> float:
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions is None: return 999999
        x, y = self.get_start_state()
        cost: float = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self) -> None:
        super().__init__()
        self.search_function = search.uniform_cost_search
        cost_fn: Callable[[tuple[int, int]], float] = lambda pos: .5 ** pos[0]
        self.search_type = lambda state: PositionSearchProblem(state, cost_fn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self) -> None:
        super().__init__()
        self.search_function = search.uniform_cost_search
        cost_fn: Callable[[tuple[int, int]], float] = lambda pos: 2 ** pos[0]
        self.search_type = lambda state: PositionSearchProblem(state, cost_fn)


def manhattan_heuristic(position: tuple[int, int], problem: PositionSearchProblem,
                        info: Optional[dict[str, Any]] = None) -> int:
    """The Manhattan distance heuristic for a PositionSearchProblem"""
    if info is None: info = {}
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclidean_heuristic(position: tuple[int, int], problem: PositionSearchProblem,
                        info: Optional[dict[str, Any]] = None) -> float:
    """The Euclidean distance heuristic for a PositionSearchProblem"""
    if info is None: info = {}
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, starting_game_state: pacman.GameState) -> None:
        """
        Stores the walls, pacman's starting position and corners.
        """
        super().__init__()
        self.walls: Grid = starting_game_state.get_walls()
        self.starting_position: tuple[int, int] = starting_game_state.get_pacman_position()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners: tuple[tuple[int, int], ...] = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not starting_game_state.has_food(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded: int = 0  # DO NOT CHANGE; Number of search nodes expanded

    def get_start_state(self) -> Any:
        """ Returns the start state (in your state space, not the full Pacman state space) """
        # TODO: Return the start state — must encode position and which corners have been visited
        #util.raise_not_defined()
        visited = [False] * len(self.corners)
        if self.starting_position in self.corners:
            index = self.corners.index(self.starting_position)
            visited[index] = True

        return (self.starting_position, tuple(visited))

    def is_goal_state(self, state: Any) -> bool:
        """Returns whether this search state is a goal state of the problem."""
        # TODO: Return True when all four corners have been visited
        #util.raise_not_defined()
        """state = (position, visited_corners_tuple) where visited_corners_tuple is a tuple of booleans"""
        if state[1] == (True, True, True, True) :
            return True

        return False # OR  return all(state[1])

    def get_successors(self, state: Any) -> list[tuple[Any, str, int]]:
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, step_cost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'step_cost'
            is the incremental cost of expanding to that successor
        """

        successors: list[tuple[Any, str, int]] = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            
            x,y = state[0]
            visited = list(state[1])
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            hits_wall = self.walls[next_x][next_y]
            if not hits_wall:
                next_state = (next_x, next_y)
                cost = 1
                if next_state in self.corners:
                    index = self.corners.index(next_state)
                    visited[index] = True
                successors.append(((next_state, tuple(visited)), action, cost))

            # TODO: Generate successor states with updated corner visitation info
            #util.raise_not_defined()

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def get_cost_of_actions(self, actions: Optional[list[str]]) -> int:
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions is None: return 999999
        x, y = self.starting_position
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def corners_heuristic(state: Any, problem: CornersProblem) -> int:
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    _corners = problem.corners  # These are the corner coordinates
    _walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)

    # TODO: Return an admissible and consistent heuristic value for the corners problem

    position, visited_corners = state
    visited_corners_list = list(visited_corners)
    un_visited_corners = []

    count = 0
    for i, item in enumerate(visited_corners_list):
        if item == False:
            corner_i = _corners[i]
            un_visited_corners.append((corner_i))
            count += 1
    if not count:
        return 0
    
    
    all_costs = []
    permutations = list(itertools.permutations(un_visited_corners))
    
    for tups in permutations:
        costs = 0
        for item in tups:
            xy1 = position
            xy2 = item
            costs += (abs(xy2[0] - xy1[0]) + abs(xy2[1] - xy1[1]))
            position = item
        all_costs  += [costs]
        
    #return 0  # Default to trivial solution
    return min(all_costs)


class AStarCornersAgent(SearchAgent):
    """A SearchAgent for CornersProblem using A* and your corners_heuristic"""

    def __init__(self) -> None:
        super().__init__()
        self.search_function = lambda prob: search.a_star_search(prob, corners_heuristic)
        self.search_type = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacman_position, food_grid ) where
      pacman_position: a tuple (x,y) of integers specifying Pacman's position
      food_grid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, starting_game_state: pacman.GameState) -> None:
        self.start: tuple[tuple[int, int], Grid] = (starting_game_state.get_pacman_position(),
                                                    starting_game_state.get_food())
        self.walls: Grid = starting_game_state.get_walls()
        self.starting_game_state: pacman.GameState = starting_game_state
        self._expanded: int = 0  # DO NOT CHANGE
        self.heuristic_info: dict[str, Any] = {}  # A dictionary for the heuristic to store information

    def get_start_state(self) -> tuple[tuple[int, int], Grid]:
        return self.start

    def is_goal_state(self, state: tuple[tuple[int, int], Grid]) -> bool:
        return state[1].count() == 0

    def get_successors(self, state: tuple[tuple[int, int], Grid]) -> list[
        tuple[tuple[tuple[int, int], Grid], str, int]]:
        """Returns successor states, the actions they require, and a cost of 1."""
        successors: list[tuple[tuple[tuple[int, int], Grid], str, int]] = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.direction_to_vector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_food = state[1].copy()
                next_food[next_x][next_y] = False
                successors.append((((next_x, next_y), next_food), direction, 1))
        return successors

    def get_cost_of_actions(self, actions: Optional[list[str]]) -> int:
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        if actions is None: return 999999
        x, y = self.get_start_state()[0]
        cost: int = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    """A SearchAgent for FoodSearchProblem using A* and your food_heuristic"""

    def __init__(self) -> None:
        super().__init__()
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem


def food_heuristic(state: tuple[tuple[int, int], Grid], _problem: FoodSearchProblem) -> int:
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacman_position, food_grid ) where food_grid is a Grid
    (see game.py) of either True or False. You can call food_grid.as_list() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristic_info that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristic_info['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristic_info['wallCount']
    """
    # TODO: Return an admissible and consistent heuristic value for the food search problem

    position, food_grid = state
    walls = _problem.walls
    list_food_pos = food_grid.as_list()
    #heuristic_dict = _problem.heuristic_info

    if len(list_food_pos) == 0:
        return 0

    cost = []
    for pos in list_food_pos:
        x1, y1 = position
        x2, y2 = pos
        if not walls[x1][y1] and not walls[x2][y2]:
            cost += [maze_distance(pos, position, _problem.starting_game_state)]
        #position = pos

    #return 0
    return max(cost)


class ClosestDotSearchAgent(SearchAgent):
    """Search for all food using a sequence of searches"""

    def __init__(self) -> None:
        super().__init__()

    def register_initial_state(self, state: pacman.GameState) -> None:
        self.actions = []
        current_state = state
        while current_state.get_food().count() > 0:
            next_path_segment = self.find_path_to_closest_dot(current_state)  # The missing piece
            self.actions += next_path_segment
            for action in next_path_segment:
                legal = current_state.get_legal_actions()
                if action not in legal:
                    t = (str(action), str(current_state))
                    raise Exception('find_path_to_closest_dot returned an illegal move: %s!\n%s' % t)
                current_state = current_state.generate_successor(0, action)
        self.action_index = 0
        print('Path found with cost %d.' % len(self.actions))

    @staticmethod
    def find_path_to_closest_dot(game_state: pacman.GameState) -> Optional[list[str]]:
        """
        Returns a path (a list of actions) to the closest dot, starting from
        game_state.
        """
        # Here are some useful elements of the start_state
        start_position = game_state.get_pacman_position()
        food = game_state.get_food()
        walls = game_state.get_walls()
        problem = AnyFoodSearchProblem(game_state)

        queue = util.Queue()
        visited = set()

        x, y = start_position

        if food[x][y]:
            return []
        
        queue.push((start_position, []))

        while not queue.is_empty():
            state, path = queue.pop()

            x, y = state

            if food[x][y]:
                return path
            
            if walls[x][y]:
                continue
            
            visited.add(state)

            for successor, action, cost in problem.get_successors(state):
                if successor not in visited:
                    queue.push((successor, path + [action]))
        # TODO: Use a search function to find a path to the closest food dot
        #util.raise_not_defined()
        return []


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the find_path_to_closest_dot
    method.
    """

    def __init__(self, game_state: pacman.GameState) -> None:
        """Stores information from the game_state.  You don't need to change this."""
        # Store the food for later reference
        self.food: Grid = game_state.get_food()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        self.cost_fn = lambda x: 1
        self._visited, self._visited_list, self._expanded = {}, [], 0  # DO NOT CHANGE

    def is_goal_state(self, state: tuple[int, int]) -> bool:
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state
        # TODO: Return True if (x, y) has food (check self.food grid)
        #util.raise_not_defined()
        return self.food[x][y]


def maze_distance(point1: tuple[int, int], point2: tuple[int, int], game_state: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The game_state can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: maze_distance( (2,4), (5,6), game_state)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = game_state.get_walls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(game_state, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
