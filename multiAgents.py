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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        flist = newFood.asList()
        foods = 0
        for food in flist:
            distance = float(util.manhattanDistance(newPos, food))
            foods = 1/distance + foods

        ghosts = 0
        for ghostState in newGhostStates:
            ghost = util.manhattanDistance(newPos, ghostState.getPosition())
            ghost = 10 * (min(1, ghost) - 1)
            ghosts = ghosts + ghost


        finalscore = score + foods + ghosts
        return finalscore

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        moves = {}
        actions = gameState.getLegalActions(0)
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
        def maxs(state, agentIndex, depth):

            agentIndex = 0
            lmoves = state.getLegalActions(agentIndex)

            if depth == self.depth or not lmoves:
                return self.evaluationFunction(state)

            maxV =  max(mins(state.generateSuccessor(agentIndex, action), \
            agentIndex + 1, depth + 1) for action in lmoves)

            return maxV

        def mins(state, agentIndex, depth):

            acount = gameState.getNumAgents()
            lmoves = state.getLegalActions(agentIndex)

            if not lmoves:
                return self.evaluationFunction(state)

            if agentIndex == acount - 1:
                minV =  min(maxs(state.generateSuccessor(agentIndex, action), \
                agentIndex,  depth) for action in lmoves)
            else:
                minV = min(mins(state.generateSuccessor(agentIndex, action), \
                agentIndex + 1, depth) for action in lmoves)

            return minV

        for action in actions:
            moves[action] = mins(gameState.generateSuccessor(0, action), 1, 1)

        return max(moves, key=moves.get)

        util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]
        util.raiseNotDefined()

    def minimax(self, gameState, index, depth,  a = -99999999, b = 99999999):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), "null")

        agents = gameState.getNumAgents()
        index = index%agents
        if index == agents - 1:
            depth = depth - 1

        if index == 0:
            return self.maxs(gameState, index, depth, a, b)
        else:
            return self.mins(gameState, index, depth, a, b)

    def maxs(self, gameState, index, depth, a, b):
        moves = []
        for action in gameState.getLegalActions(index):
            maxV = self.minimax(gameState.generateSuccessor(index, action), index + 1, depth, a, b)[0]
            moves.append((maxV, action))
            if maxV > b:
                return (maxV, action)
            a = max(a, maxV)
        return max(moves)

    def mins(self, gameState, index, depth, a, b):
        moves = []
        for action in gameState.getLegalActions(index):
            minV = self.minimax(gameState.generateSuccessor(index, action), index + 1, depth, a, b)[0]
            moves.append((minV, action))
            if minV < a:
                return (minV, action)
            b = min(b, minV)
        return min(moves)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expected(self, gameState, index, depth):
        gmoves = gameState.getLegalActions(index)
        cost = []

        if gameState.isLose() or not gmoves:
            return self.evaluationFunction(gameState), None

        next = [gameState.generateSuccessor(index, action) for action in gmoves]

        for successor in next:
            if index == gameState.getNumAgents() - 1:
                cost.append(self.maxs(successor, depth + 1))
            else:
                cost.append(self.expected(successor, index + 1, depth))

        mean = sum(map(lambda x: float(x[0]) / len(cost), cost))
        return mean, None


    def maxs(self, gameState, depth):
        lmoves = gameState.getLegalActions(0)
        cost = []

        if gameState.isWin() or depth > self.depth or not lmoves:
            return self.evaluationFunction(gameState), None

        for action in lmoves:
            next = gameState.generateSuccessor(0, action)
            cost.append((self.expected(next, 1, depth)[0], action))

        return max(cost)


    def getAction(self, gameState):


        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxs(gameState, 1)[1]




        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood()
    flist = food.asList()
    fdistance = []

    gstate = currentGameState.getGhostStates()
    gval = 1.0
    fval = 1.0
    sgval = 5.0


    for state in gstate:

        distance = manhattanDistance(pos, state.getPosition())
        if distance > 0:
            if state.scaredTimer > 0:
                score = score + sgval / distance
            else:
                score = score - gval / distance


    for state in flist:
        fdistance.append(manhattanDistance(pos, state))

    if len(fdistance) is not 0:
        score = score + fval / min(fdistance)

    return score

    util.raiseNotDefined()




    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
