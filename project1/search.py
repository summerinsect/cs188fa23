# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    visit = set()
    prev = dict()
    start = problem.getStartState()
    fringe.push(start)
    prev[start] = (start, -1)
    while not fringe.isEmpty():
        current = fringe.pop()
        if current in visit:
            continue
        visit.add(current)
        if problem.isGoalState(current):
            path = list()
            last = prev[current][0]
            while not last == current:
                path.append(prev[current][1])
                current = last
                last = prev[current][0]
            path.reverse()
            return path
        successors = problem.getSuccessors(current)
        for (next, action, cost) in successors:
            if not next in visit:
                fringe.push(next)
                prev[next] = (current, action)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    visit = set()
    prev = dict()
    start = problem.getStartState()
    fringe.push(start)
    visit.add(start)
    prev[start] = (start, -1)
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.isGoalState(current):
            path = list()
            last = prev[current][0]
            while not last == current:
                path.append(prev[current][1])
                current = last
                last = prev[current][0]
            path.reverse()
            return path
        successors = problem.getSuccessors(current)
        for (next, action, cost) in successors:
            if not next in visit:
                fringe.push(next)
                visit.add(next)
                prev[next] = (current, action)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    visit = set()
    prev = dict()
    prior = dict()
    start = problem.getStartState()
    fringe.push(start, 0)
    prev[start] = (start, -1)
    prior[start] = 0
    while not fringe.isEmpty():
        current = fringe.pop()
        if current in visit:
            continue
        visit.add(current)
        if problem.isGoalState(current):
            path = list()
            last = prev[current][0]
            while not last == current:
                path.append(prev[current][1])
                current = last
                last = prev[current][0]
            path.reverse()
            return path
        successors = problem.getSuccessors(current)
        for (next, action, cost) in successors:
            if not next in visit:
                value = prior[current] + cost
                if next not in prior.keys():
                    prior[next] = 1e60
                if prior[next] > value:
                    prior[next] = value
                    fringe.push(next, prior[next])
                    prev[next] = (current, action)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    g, h = dict(), dict()
    visit = set()
    prev = dict()

    def getValueH(state):
        if state not in h.keys():
            h[state] = heuristic(state, problem)
        return h[state]

    def getValueG(state):
        if state not in g.keys():
            g[state] = 1e60
        return g[state]

    def getValueF(state):
        return getValueG(state) + getValueH(state)

    start = problem.getStartState()
    g[start] = 0
    fringe.push(start, getValueF(start))
    prev[start] = (start, -1)
    while not fringe.isEmpty():
        current = fringe.pop()
        if current in visit:
            continue
        visit.add(current)
        if problem.isGoalState(current):
            path = list()
            last = prev[current][0]
            while not last == current:
                path.append(prev[current][1])
                current = last
                last = prev[current][0]
            path.reverse()
            return path
        successors = problem.getSuccessors(current)
        for (next, action, cost) in successors:
            if not next in visit:
                value = getValueG(current) + cost
                if getValueG(next) > value:
                    g[next] = value
                    fringe.push(next, getValueF(next))
                    prev[next] = (current, action)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
