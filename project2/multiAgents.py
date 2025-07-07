# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPacman = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        stateValue = 0
        if newPacman in currentGameState.getFood().asList():
            stateValue += 100

        for food in newFood:
            "*** stateValue += .8 ** util.manhattanDistance(newPacman, food) ***"
            stateValue += 1. / util.manhattanDistance(newPacman, food)
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            dis = util.manhattanDistance(newPacman, ghostPosition)
            if dis < 0.7:
                return -1e30
            if ghost.scaredTimer <= 10:
                stateValue -= 10. / dis
        return stateValue
        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimaxSearch(agentIndex, gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), 0)
            legalMoves = gameState.getLegalActions(agentIndex)
            nextIndex, nextDepth = agentIndex + 1, depth
            if nextIndex == gameState.getNumAgents():
                nextIndex, nextDepth = 0, depth + 1
            bestScore, bestChoice = 1e30, 0
            if agentIndex == 0:
                bestScore = -1e30
            for action in legalMoves:
                successorState = gameState.generateSuccessor(agentIndex, action)
                score, _ = minimaxSearch(nextIndex, successorState, nextDepth)
                if agentIndex == 0 and score > bestScore:
                    bestScore, bestChoice = score, action
                if agentIndex != 0 and score < bestScore:
                    bestScore, bestChoice = score, action
            return (bestScore, bestChoice)
        return minimaxSearch(0, gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimaxSearch(agentIndex, gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), 0)
            legalMoves = gameState.getLegalActions(agentIndex)
            nextIndex, nextDepth = agentIndex + 1, depth
            if nextIndex == gameState.getNumAgents():
                nextIndex, nextDepth = 0, depth + 1
            bestScore, bestChoice = 1e50, 0
            if agentIndex == 0:
                bestScore = -1e50
            for action in legalMoves:
                successorState = gameState.generateSuccessor(agentIndex, action)
                score, _ = minimaxSearch(nextIndex, successorState, alpha, beta, nextDepth)
                if agentIndex == 0 and score > bestScore:
                    bestScore, bestChoice = score, action
                    if bestScore > beta:
                        return (bestScore, bestChoice)
                    alpha = max(alpha, bestScore)
                if agentIndex != 0 and score < bestScore:
                    bestScore, bestChoice = score, action
                    if bestScore < alpha:
                        return (bestScore, bestChoice)
                    beta = min(beta, bestScore)
            return (bestScore, bestChoice)
        return minimaxSearch(0, gameState, -1e50, 1e50, 0)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimaxSearch(agentIndex, gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), 0)
            legalMoves = gameState.getLegalActions(agentIndex)
            nextIndex, nextDepth = agentIndex + 1, depth
            if nextIndex == gameState.getNumAgents():
                nextIndex, nextDepth = 0, depth + 1
            if agentIndex == 0:
                bestScore, bestChoice = -1e30, 0
                for action in legalMoves:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score, _ = expectimaxSearch(nextIndex, successorState, nextDepth)
                    if score > bestScore:
                        bestScore, bestChoice = score, action
                return (bestScore, bestChoice)
            else:
                averageScore = 0
                for action in legalMoves:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score, _ = expectimaxSearch(nextIndex, successorState, nextDepth)
                    averageScore += score
                return (averageScore / len(legalMoves), 0)
        return expectimaxSearch(0, gameState, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -1e30
    pacman = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood().asList()
    capsulesPosition = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    stateValue = 10000 - 10 * len(foodPosition) - 5 * len(capsulesPosition)

    distances = dict()
    def prework():
        fringe = util.Queue()
        fringe.push(pacman)
        distances[pacman] = 0
        walls = currentGameState.getWalls()
        while not fringe.isEmpty():
            currentx, currenty = fringe.pop()
            current = (currentx, currenty)
            for (dx, dy) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nextx, nexty = currentx + dx, currenty + dy
                next = (nextx, nexty)
                if not walls[nextx][nexty] and next not in distances.keys():
                    distances[next] = distances[current] + 1
                    fringe.push(next)
    prework()

    for food in foodPosition:
        stateValue += .8 ** distances[food]
        "*** stateValue += 1. / distances[food] ***"
    for capsules in capsulesPosition:
        stateValue += .85 ** distances[capsules]
    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        ghostx, ghosty = int(ghostPosition[0]), int(ghostPosition[1])
        dis = distances[(ghostx, ghosty)]
        if ghost.scaredTimer <= 3:
            stateValue -= 1. / dis
        else:
            stateValue += 100 * 0.5 ** dis
    return stateValue

# Abbreviation
better = betterEvaluationFunction
