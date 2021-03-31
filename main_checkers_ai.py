"""
CS4701 Practicum in AI   
Project: Checkers AI with minimax algorithm
Team members: Shunqi Mao, Pu Zhao
"""

import copy
import math
import random
import datetime
import tkinter
import _thread

class AIPlayer():
    def __init__(self, game, difficulty):
        self.game = game
        self.difficulty = difficulty

    def getNextMove(self):
        if self.difficulty == 1:
            return self.getNextMoveEasy()
        elif self.difficulty == 2:
            return self.getNextMoveMedium()
        elif self.difficulty == 3:
            return self.getNextMoveHard()
        else:
            return self.getNextMoveInfernal()

    # Simple AI, returns a random legal move
    def getNextMoveEasy(self):
        state = AIGameState(self.game)
        moves = state.getActions(False)
        index = random.randrange(len(moves))
        chosenMove = moves[index]
        return chosenMove[0], chosenMove[1], chosenMove[2], chosenMove[3]

    # Hard AI, returns the move found by alpha-beta search with depth limit 5
    def getNextMoveMedium(self):
        state = AIGameState(self.game)
        nextMove = self.alphaBetaSearch(state, 5)
        return nextMove[0], nextMove[1], nextMove[2], nextMove[3]

    # Hard AI, returns the move found by alpha-beta search with depth limit 10
    def getNextMoveHard(self):
        state = AIGameState(self.game)
        nextMove = self.alphaBetaSearch(state, 10)
        return nextMove[0], nextMove[1], nextMove[2], nextMove[3]
    
    # Infernal AI, returns the best move found by alpha-beta search
    def getNextMoveInfernal(self):
        state = AIGameState(self.game)
        depthLimit = self.computeDepthLimit(state)
        nextMove = self.alphaBetaSearch(state, depthLimit)
        return nextMove[0], nextMove[1], nextMove[2], nextMove[3]

    # Dynamically compute depth limit
    # Fewer checkers we have, deeper level we can search
    def computeDepthLimit(self, state):
        numcheckers = len(state.AIpieces) + len(state.humanPieces)
        return 26 - numcheckers

    def alphaBetaSearch(self, state, depthLimit):
        # collect statistics for the search
        self.currDepth = 0
        self.maxDepth = 0
        self.numNodes = 0
        self.maxPruning = 0
        self.minPruning = 0

        self.bestMove = []
        self.depthLimit = depthLimit

        starttime = datetime.datetime.now()
        v = self.maxValue(state, -1000, 1000, self.depthLimit)

        # print statistics for the search

        print("Time = " + str(datetime.datetime.now() - starttime))
        print("selected value " + str(v))
        print("(1) max depth of the tree = {0:d}".format(self.maxDepth))
        print("(2) total number of nodes generated = {0:d}".format(self.numNodes))
        print("(3) number of times pruning occurred in the MAX-VALUE() = {0:d}".format(self.maxPruning))
        print("(4) number of times pruning occurred in the MIN-VALUE() = {0:d}".format(self.minPruning))

        return self.bestMove

    # For AI player (MAX AGENT)
    def maxValue(self, state, alpha, beta, depthLimit):
        if state.terminalTest():
            return state.computeUtilityValue()
        if depthLimit == 0:
            return state.computeHeuristic()

        # update statistics for the search
        self.currDepth += 1
        self.maxDepth = max(self.maxDepth, self.currDepth)
        self.numNodes += 1

        v = -math.inf
        for a in state.getActions(False):
            # return captured checker if it is a capture move
            captured = state.applyAction(a)
            # state.printBoard()
            if state.humanCanContinue():
                next = self.minValue(state, alpha, beta, depthLimit - 1)
            else:  # human cannot move, AI gets one more move
                next = self.maxValue(state, alpha, beta, depthLimit - 1)
            if next > v:
                v = next
                # Keep track of the best move so far at the top level
                if depthLimit == self.depthLimit:
                    self.bestMove = a
            state.revoke(a, captured)

            # alpha-beta max pruning
            if v >= beta:
                self.maxPruning += 1
                self.currDepth -= 1
                return v
            alpha = max(alpha, v)

        self.currDepth -= 1

        return v

    # For human player (MIN AGENT)
    def minValue(self, state, alpha, beta, depthLimit):
        if state.terminalTest():
            return state.computeUtilityValue()
        if depthLimit == 0:
            return state.computeHeuristic()

        # update statistics for the search
        self.currDepth += 1
        self.maxDepth = max(self.maxDepth, self.currDepth)
        self.numNodes += 1

        v = math.inf
        for a in state.getActions(True):
            captured = state.applyAction(a)
            if state.AICanContinue():
                next = self.maxValue(state, alpha, beta, depthLimit - 1)
            else:  # AI cannot move, human gets one more move
                next = self.minValue(state, alpha, beta, depthLimit - 1)
            if next < v:
                v = next
            state.revoke(a, captured)

            #alpha-beta min pruning
            if v <= alpha:
                self.minPruning += 1
                self.currDepth -= 1
                return v
            beta = min(beta, v)

        self.currDepth -= 1
        return v

# a class for AI to simulate game state
class AIGameState():
    def __init__(self, game):
        self.board = copy.deepcopy(game.getBoard())

        self.AIpieces = set()
        for checker in game.opponentPieces:
            self.AIpieces.add(checker)
        self.humanPieces = set()
        for checker in game.playerPieces:
            self.humanPieces.add(checker)
        self.checkerPositions = {}
        for checker, position in game.checkerPositions.items():
            self.checkerPositions[checker] = position

    # Check if the human player can cantinue.
    def humanCanContinue(self):
        directions = [[-1, -1], [-1, 1], [-2, -2], [-2, 2]]
        for checker in self.humanPieces:
            position = self.checkerPositions[checker]
            row = position[0]
            col = position[1]
            for dir in directions:
                if self.isValidMove(row, col, row + dir[0], col + dir[1], True):
                    return True
        return False

    # Check if the AI player can cantinue.
    def AICanContinue(self):
        directions = [[1, -1], [1, 1], [2, -2], [2, 2]]
        for checker in self.AIpieces:
            position = self.checkerPositions[checker]
            row = position[0]
            col = position[1]
            for dir in directions:
                if self.isValidMove(row, col, row + dir[0], col + dir[1], False):
                    return True
        return False

    # Neither player can can continue, thus game over
    def terminalTest(self):
        if len(self.humanPieces) == 0 or len(self.AIpieces) == 0:
            return True
        else:
            return (not self.AICanContinue()) and (not self.humanCanContinue())

    # Check if current move is valid
    def isValidMove(self, oldrow, oldcol, row, col, humanTurn):
        # invalid index
        if oldrow < 0 or oldrow > 7 or oldcol < 0 or oldcol > 7 \
                or row < 0 or row > 7 or col < 0 or col > 7:
            return False
        # No checker exists in original position
        if self.board[oldrow][oldcol] == 0:
            return False
        # Another checker exists in destination position
        if self.board[row][col] != 0:
            return False

        # human player's turn
        if humanTurn:
            if row - oldrow == -1:   # regular move
                return abs(col - oldcol) == 1
            elif row - oldrow == -2:  # capture move
                return (col - oldcol == -2 and self.board[row+1][col+1] < 0) \
                       or (col - oldcol == 2 and self.board[row+1][col-1] < 0)
            else:
                return False
        # opponent's turn
        else:
            if row - oldrow == 1:   # regular move
                return abs(col - oldcol) == 1
            elif row - oldrow == 2: # capture move
                # / direction or \ direction
                return (col - oldcol == -2 and self.board[row-1][col+1] > 0) \
                       or (col - oldcol == 2 and self.board[row-1][col-1] > 0)
            else:
                return False

    # compute utility value of terminal state
    # utility value = difference in # of checkers * 500 + # of AI checkers * 50
    # utility value has larger weights so that is it preferred over heuristic values
    def computeUtilityValue(self):
        utility = (len(self.AIpieces) - len(self.humanPieces)) * 500 \
                  + len(self.AIpieces) * 50
        return utility

    # compute heuristic value of a non-terminal state
    # heuristic value = diff in # of checkers * 50 + # of safe checkers * 10 + # of AI checkers
    def computeHeuristic(self):
        heurisitc = (len(self.AIpieces) - len(self.humanPieces)) * 50 \
                    + self.countSafeAIpieces() * 10 + len(self.AIpieces)
        return heurisitc

    # Count the number of safe AI checker.
    # A safe AI checker is one checker that no opponent can capture.
    def countSafeAIpieces(self):
        count = 0
        for AIchecker in self.AIpieces:
            AIrow = self.checkerPositions[AIchecker][0]
            AIcol = self.checkerPositions[AIchecker][1]
            safe = True
            if not (AIcol == 0 or AIcol == len(self.board[0])):
                # checkers near the boundaries are safe
                for humanchecker in self.humanPieces:
                    if AIrow < self.checkerPositions[humanchecker][0]:
                        safe = False
                        break
            if safe:
                count += 1
        return count

    # get all possible actions for the current player
    def getActions(self, humanTurn):
        if humanTurn:
            checkers = self.humanPieces
            regularDirs = [[-1, -1], [-1, 1]]
            captureDirs = [[-2, -2], [-2, 2]]
        else:
            checkers = self.AIpieces
            regularDirs = [[1, -1], [1, 1]]
            captureDirs = [[2, -2], [2, 2]]

        regularMoves = []
        captureMoves = []
        for checker in checkers:
            oldrow = self.checkerPositions[checker][0]
            oldcol = self.checkerPositions[checker][1]
            for dir in regularDirs:
                if self.isValidMove(oldrow, oldcol, oldrow+dir[0], oldcol+dir[1], humanTurn):
                    regularMoves.append([oldrow, oldcol, oldrow+dir[0], oldcol+dir[1]])
            for dir in captureDirs:
                if self.isValidMove(oldrow, oldcol, oldrow+dir[0], oldcol+dir[1], humanTurn):
                    captureMoves.append([oldrow, oldcol, oldrow+dir[0], oldcol+dir[1]])

        # must take capture move if possible
        if captureMoves:
            return captureMoves
        else:
            return regularMoves

    # Apply given action to the game board.
    def applyAction(self, action):
        oldrow = action[0]
        oldcol = action[1]
        row = action[2]
        col = action[3]
        captured = 0

        # move the checker
        toMove = self.board[oldrow][oldcol]
        self.checkerPositions[toMove] = (row, col)
        self.board[row][col] = toMove
        self.board[oldrow][oldcol] = 0

        # capture move, remove captured checker
        if abs(oldrow - row) == 2:
            captured = self.board[(oldrow + row) // 2][(oldcol + col) // 2]
            if captured > 0:
                self.humanPieces.remove(captured)
            else:
                self.AIpieces.remove(captured)
            self.board[(oldrow + row) // 2][(oldcol + col) // 2] = 0
            self.checkerPositions.pop(captured, None)

        return captured

    # Reset given action to the game board. Restored captured checker if any.
    def revoke(self, action, captured):
        oldrow = action[0]
        oldcol = action[1]
        row = action[2]
        col = action[3]

        # move the checker
        toMove = self.board[row][col]
        self.checkerPositions[toMove] = (oldrow, oldcol)
        self.board[oldrow][oldcol] = toMove
        self.board[row][col] = 0

        # capture move, remove captured checker
        if abs(oldrow - row) == 2:
            if captured > 0:
                self.humanPieces.add(captured)
            else:
                self.AIpieces.add(captured)
            self.board[(oldrow + row) // 2][(oldcol + col) // 2] = captured
            self.checkerPositions[captured] = ((oldrow + row) // 2, (oldcol + col) // 2)

    def printBoard(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                check = self.board[i][j]
                if (check < 0):
                    print(check,end=' ')
                else:
                    print(' ' + str(check),end=' ')

            print()
        print('------------------------')

class CheckerGame():
    def __init__(self):
        self.lock = _thread.allocate_lock()
        self.board = self.initBoard()
        self.playerTurn = self.whoGoFirst()
        self.difficulty = self.getDifficulty()
        self.AIPlayer = AIPlayer(self, self.difficulty)
        self.GUI = BoardGUI(self)

        # AI goes first
        if not self.isPlayerTurn():
            _thread.start_new_thread(self.AIMakeMove, ())

        self.GUI.startGUI()

    # Let player decide to go first or second
    def whoGoFirst(self):
        ans = input("Do you want to go first? (Y/N) ")
        return ans == "Y" or ans == "y"

    # Let player decide level of difficulty
    def getDifficulty(self):
        ans = eval(input("What level of difficulty? (1 Easy, 2 Medium, 3 Hard, 4 Infernal) "))
        while not (ans == 1 or ans == 2 or ans == 3 or ans == 4):
            print("Invalid input, please enter a value between 1 and 4")
            ans = eval(input("What level of difficulty? (1 Easy, 2 Medium, 3 Hard) "))
        return ans

    # This function initializes the game board.
    # Each checker has a label. Positive checkers for the player,
    # and negative checkers for the opponent.
    def initBoard(self):
        board = [[0]*8 for _ in range(8)]
        self.playerPieces = set()
        self.opponentPieces = set()
        self.checkerPositions = {}
        for i in range(8):
            self.playerPieces.add(i + 1)
            self.opponentPieces.add(-(i + 1))
            if i % 2 == 0:
                board[1][i] = -(i + 1)
                board[7][i] = i + 1
                self.checkerPositions[-(i + 1)] = (1, i)
                self.checkerPositions[i + 1] = (7, i)
            else:
                board[0][i] = -(i + 1)
                board[6][i] = i + 1
                self.checkerPositions[-(i + 1)] = (0, i)
                self.checkerPositions[i + 1] = (6, i)

        self.boardUpdated = True

        return board

    def getBoard(self):
        return self.board

    def printBoard(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                check = self.board[i][j]
                if (check < 0):
                    print(check,end=' ')
                else:
                    print(' ' + str(check),end=' ')

            print()

    def isBoardUpdated(self):
        return self.boardUpdated

    def setBoardUpdated(self):
        self.lock.acquire()
        self.boardUpdated = True
        self.lock.release()

    def completeBoardUpdate(self):
        self.lock.acquire()
        self.boardUpdated = False
        self.lock.release()

    def isPlayerTurn(self):
        return self.playerTurn

    # Switch turns between player and opponent.
    # If one of them has no legal moves, the other can keep playing
    def changePlayerTurn(self):
        if self.playerTurn and self.opponentCanContinue():
            self.playerTurn = False
        elif not self.playerTurn and self.playerCanContinue():
            self.playerTurn = True

    # apply the given move in the game
    def move(self, oldrow, oldcol, row, col):
        if not self.isValidMove(oldrow, oldcol, row, col, self.playerTurn):
            return

        # human player can only choose from the possible actions
        if self.playerTurn and not ([oldrow, oldcol, row, col] in self.getPossiblePlayerActions()):
            return

        self.makeMove(oldrow, oldcol, row, col)
        _thread.start_new_thread(self.next, ())

    # update game state
    def next(self):
        if self.isGameOver():
            self.getGameSummary()
            return
        self.changePlayerTurn()
        if self.playerTurn:     # let player keep going
            return
        else:                   # AI's turn
            self.AIMakeMove()

    # Temporarily Pause GUI and ask AI player to make next move.
    def AIMakeMove(self):
        self.GUI.pauseGUI()
        oldrow, oldcol, row, col = self.AIPlayer.getNextMove()
        self.move(oldrow, oldcol, row, col)
        self.GUI.resumeGUI()

    # update checker position
    def makeMove(self, oldrow, oldcol, row, col):
        toMove = self.board[oldrow][oldcol]
        self.checkerPositions[toMove] = (row, col)

        # move the checker
        self.board[row][col] = self.board[oldrow][oldcol]
        self.board[oldrow][oldcol] = 0

        # capture move, remove captured checker
        if abs(oldrow - row) == 2:
            toRemove = self.board[(oldrow + row) // 2][(oldcol + col) // 2]
            if toRemove > 0:
                self.playerPieces.remove(toRemove)
            else:
                self.opponentPieces.remove(toRemove)
            self.board[(oldrow + row) // 2][(oldcol + col) // 2] = 0
            self.checkerPositions.pop(toRemove, None)

        self.setBoardUpdated()

    # Get all possible moves for the current player
    def getPossiblePlayerActions(self):
        checkers = self.playerPieces
        regularDirs = [[-1, -1], [-1, 1]]
        captureDirs = [[-2, -2], [-2, 2]]

        regularMoves = []
        captureMoves = []
        for checker in checkers:
            oldrow = self.checkerPositions[checker][0]
            oldcol = self.checkerPositions[checker][1]
            for dir in regularDirs:
                if self.isValidMove(oldrow, oldcol, oldrow+dir[0], oldcol+dir[1], True):
                    regularMoves.append([oldrow, oldcol, oldrow+dir[0], oldcol+dir[1]])
            for dir in captureDirs:
                if self.isValidMove(oldrow, oldcol, oldrow+dir[0], oldcol+dir[1], True):
                    captureMoves.append([oldrow, oldcol, oldrow+dir[0], oldcol+dir[1]])

        # must take capture move if possible
        if captureMoves:
            return captureMoves
        else:
            return regularMoves

    # check if the given move is valid for the current player
    def isValidMove(self, oldrow, oldcol, row, col, playerTurn):
        # invalid index
        if oldrow < 0 or oldrow > 7 or oldcol < 0 or oldcol > 7 \
                or row < 0 or row > 7 or col < 0 or col > 7:
            return False
        # No checker exists in original position
        if self.board[oldrow][oldcol] == 0:
            return False
        # Another checker exists in destination position
        if self.board[row][col] != 0:
            return False

        # player's turn
        if playerTurn:
            if row - oldrow == -1:   # regular move
                return abs(col - oldcol) == 1
            elif row - oldrow == -2:  # capture move
                #  \ direction or / direction
                return (col - oldcol == -2 and self.board[row+1][col+1] < 0) \
                       or (col - oldcol == 2 and self.board[row+1][col-1] < 0)
            else:
                return False
        # opponent's turn
        else:
            if row - oldrow == 1:   # regular move
                return abs(col - oldcol) == 1
            elif row - oldrow == 2: # capture move
                # / direction or \ direction
                return (col - oldcol == -2 and self.board[row-1][col+1] > 0) \
                       or (col - oldcol == 2 and self.board[row-1][col-1] > 0)
            else:
                return False

    # Check if the player can cantinue
    def playerCanContinue(self):
        directions = [[-1, -1], [-1, 1], [-2, -2], [-2, 2]]
        for checker in self.playerPieces:
            position = self.checkerPositions[checker]
            row = position[0]
            col = position[1]
            for dir in directions:
                if self.isValidMove(row, col, row + dir[0], col + dir[1], True):
                    return True
        return False

    # Check if the opponent can cantinue
    def opponentCanContinue(self):
        directions = [[1, -1], [1, 1], [2, -2], [2, 2]]
        for checker in self.opponentPieces:
            position = self.checkerPositions[checker]
            row = position[0]
            col = position[1]
            for dir in directions:
                if self.isValidMove(row, col, row + dir[0], col + dir[1], False):
                    return True
        return False

    # Neither player can can continue, thus game over
    def isGameOver(self):
        if len(self.playerPieces) == 0 or len(self.opponentPieces) == 0:
            return True
        else:
            return (not self.playerCanContinue()) and (not self.opponentCanContinue())

    def getGameSummary(self):
        self.GUI.pauseGUI()
        print("Game Over!")
        playerNum = len(self.playerPieces)
        opponentNum = len(self.opponentPieces)
        if (playerNum > opponentNum):
            print("Player won by {0:d} checkers! Congratulation!".format(playerNum - opponentNum))
        elif (playerNum < opponentNum):
            print("Computer won by {0:d} checkers! Try again!".format(opponentNum - playerNum))
        else:
            print("It is a draw! Try again!")

class BoardGUI():
    def __init__(self, game):
        # Initialize parameters
        self.game = game
        self.ROWS = 8
        self.COLS = 8
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 800
        self.col_width = self.WINDOW_WIDTH / self.COLS
        self.row_height = self.WINDOW_HEIGHT / self.ROWS

        # Initialize GUI
        self.initBoard()

    def initBoard(self):
        self.root = tkinter.Tk()
        self.c = tkinter.Canvas(self.root, width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT,
                                borderwidth=5, background='white')
        self.c.pack()
        self.board = [[0]*self.COLS for _ in range(self.ROWS)]
        self.tiles = [[None for _ in range(self.COLS)] for _ in range(self.ROWS)]

        # Print dark square
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.c.create_rectangle(i * self.row_height, j * self.col_width,
                                        (i+1) * self.row_height, (j+1) * self.col_width, fill="gray", outline="gray")

        # Print grid lines
        for i in range(8):
            self.c.create_line(0, self.row_height * i, self.WINDOW_WIDTH, self.row_height * i, width=2)
            self.c.create_line(self.col_width * i, 0, self.col_width * i, self.WINDOW_HEIGHT, width=2)

        # Place checks on the board
        self.updateBoard()

        # Initialize parameters
        self.checkerSelected = False
        self.clickData = {"row": 0, "col": 0, "checker": None}

        # Register callback function for mouse clicks
        self.c.bind("<Button-1>", self.processClick)

        # make GUI updates board every second
        self.root.after(1000, self.updateBoard)


    def startGUI(self):
        self.root.mainloop()

    def pauseGUI(self):
        self.c.bind("<Button-1>", '')

    def resumeGUI(self):
        self.c.bind("<Button-1>", self.processClick)

    # Update the positions of checkers
    def updateBoard(self):
        if self.game.isBoardUpdated():
            newBoard = self.game.getBoard()
            for i in range(len(self.board)):
                for j in range(len(self.board[0])):
                    if self.board[i][j] != newBoard[i][j]:
                        self.board[i][j] = newBoard[i][j]
                        self.c.delete(self.tiles[i][j])
                        self.tiles[i][j] = None

                        # choose different color for different player's checkers
                        if newBoard[i][j] < 0:
                            self.tiles[i][j] = self.c.create_oval(j*self.col_width+10, i*self.row_height+10,
                                                              (j+1)*self.col_width-10, (i+1)*self.row_height-10,
                                                              fill="black")
                        elif newBoard[i][j] > 0:
                            self.tiles[i][j] = self.c.create_oval(j*self.col_width+10, i*self.row_height+10,
                                                                  (j+1)*self.col_width-10, (i+1)*self.row_height-10,
                                                                  fill="red")
                        else:  # no checker
                            continue

                        # raise the tiles to highest layer
                        self.c.tag_raise(self.tiles[i][j])

            # tell game logic that GUI has updated the board
            self.game.completeBoardUpdate()

        # make GUI updates board every second
        self.root.after(1000, self.updateBoard)

    # this function checks if the checker belongs to the current player
    def isCurrentPlayerChecker(self, row, col):
        return self.game.isPlayerTurn() == (self.board[row][col] > 0)

    # callback function that process user's mouse clicks
    def processClick(self, event):
        col = int(event.x // self.col_width)
        row = int(event.y // self.row_height)

        # If there is no checker being selected
        if not self.checkerSelected:
            # there exists a checker at the clicked position
            # and the checker belongs to the current player
            if self.board[row][col] != 0 and self.isCurrentPlayerChecker(row, col):
                self.clickData["row"] = row
                self.clickData["col"] = col
                self.clickData["color"] = self.c.itemcget(self.tiles[row][col], 'fill')

                # replace clicked checker with a temporary checker
                self.c.delete(self.tiles[row][col])
                self.tiles[row][col] = self.c.create_oval(col*self.col_width+10, row*self.row_height+10,
                                                         (col+1)*self.col_width-10, (row+1)*self.row_height-10,
                                                          fill="green")
                self.checkerSelected = True

            else: # no checker at the clicked postion
                return

        else: # There is a checker being selected
            # First reset the board
            oldrow = self.clickData["row"]
            oldcol = self.clickData["col"]
            self.c.delete(self.tiles[oldrow][oldcol])
            self.tiles[oldrow][oldcol] = self.c.create_oval(oldcol*self.col_width+10, oldrow*self.row_height+10,
                                                            (oldcol+1)*self.col_width-10, (oldrow+1)*self.row_height-10,
                                                            fill=self.clickData["color"])

            # If the destination leads to a legal move
            self.game.move(self.clickData["row"], self.clickData["col"],row, col)
            self.checkerSelected = False


game = CheckerGame()