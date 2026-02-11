## Adverserial Sudoku AI

### The Task
The task at hand is to build an artificial intelligence player for a two-player competitive version of Sudoku. Unlike traditional Sudoku where we simply fill in numbers to complete a puzzle, this is a game between two players who take turns placing numbers on the board, competing to score the most points. The AI must decide which move to make on each turn, trying to maximize its own score while minimizing the opponent's scoring opportunities.

The assignment required implementing several game-playing algorithms that computer scientists have developed over decades for playing strategic games. These include the minimax algorithm, alpha-beta pruning, iterative deepening, and transposition tables. The challenge was not just understanding these algorithms but combining them effectively to create an AI that can make good decisions within a strict time limit.


### The Game Board and Basic Rules
The game is played on a Sudoku board, which is a grid divided into smaller rectangular regions called blocks. The most common Sudoku is a nine by nine grid with three by three blocks, but this game supports various sizes. For example, a two by two block configuration creates a four by four grid, while a two by three block configuration creates a six by six grid. The board can start empty or partially filled with some numbers already placed.

The fundamental rule of Sudoku, which we call Constraint C0, states that every row, every column, and every block must contain unique values. If the grid is nine by nine, each row must contain the numbers one through nine exactly once, each column must contain one through nine exactly once, and each three by three block must also contain one through nine exactly once. This constraint must never be violated during gameplay.

### How this Two-Player Game Works
Two players alternate turns throughout the game. On each turn, a player chooses an empty cell and places a valid number in it. The game continues until the board is completely filled or neither player can make a valid move.

The scoring system rewards players for completing regions. When we place a number that completes a row, meaning it was the last empty cell in that row, we earn points. The same applies to completing columns and blocks. A single move can complete zero, one, two, or even three regions simultaneously if the cell happens to be the last empty cell in its row, its column, and its block all at once.

The points awarded follow a specific pattern. Completing zero regions gives zero points. Completing one region gives one point. Completing two regions gives three points. Completing three regions gives seven points. Notice that the points increase more than linearly, making it very valuable to complete multiple regions with a single move.


### The Oracle and Taboo Moves
The game includes a special mechanism called the oracle, which is an external program that can solve Sudoku puzzles. After each move, the oracle checks whether the puzzle can still be solved. If our move makes the puzzle unsolvable, meaning there is no way to fill in the remaining cells without violating Sudoku rules, your move is declared taboo.
When a move is taboo, it is not actually played on the board, we receive zero points, and the move is added to a list of forbidden moves. Neither player can attempt that same move again. This prevents players from intentionally or accidentally breaking the puzzle.
There is also a critical rule about illegal moves. If we attempt a move that violates the basic Sudoku constraint, meaning you place a number that already exists in that row, column, or block, we lose the game immediately. Our opponent wins regardless of the current scores. This makes it essential that your AI never proposes such moves.


### The Allowed Squares System
Depending on the game mode, players may not be able to play on every empty square. The game supports different modes like classic, rows, border, and random. In classic mode, both players can play anywhere on the board. In other modes, each player starts with a designated set of squares they can play on.
The expansion rule adds complexity to non-classic modes. After we make a move, all squares adjacent to your move, including diagonally adjacent squares, become available to us. This means the set of squares you can play on grows as we make moves, spreading outward from our initial territory.


### The Time Constraint Challenge
Perhaps the most important constraint to understand is the time limit. When it is our turn, the game framework starts your AI in a separate process and gives it a fixed amount of time to think, typically one second or a few seconds. When time runs out, the framework does not politely ask your AI to stop. Instead, it forcefully terminates our process immediately.
This means our AI might be killed at any moment during its computation. If we have not proposed any move before being killed, we lose. If we proposed a move early but found a better one later that we never got to propose, we are stuck with the earlier, inferior move.
This constraint fundamentally shapes how the AI must be designed. It must be an anytime algorithm, meaning it must always have a valid move ready and should continuously improve its answer as long as time permits.


### Why We Cannot Use the Oracle
The example players provided with the framework, such as the greedy player and random player, use the oracle to check which moves are valid. This might seem convenient, but we are explicitly forbidden from using the oracle in our AI. The oracle is only for the framework to validate moves after they are made.
This restriction exists because calling the oracle is computationally expensive and would be considered cheating in a sense. Our AI must determine valid moves using only its own logic and the current game state. This is why implementing proper constraint checking is crucial.


### The Solution Approach
Our solution combines several classical game-playing techniques to create an effective AI. Each technique addresses a specific challenge, and together they form a cohesive system.

#### The Minimax Algorithm
The minimax algorithm is the foundation of our AI. It treats the game as a tree where each node represents a game state and each branch represents a possible move. The algorithm assumes both players play optimally, meaning you will always make the best move for yourself, and your opponent will always make the best move for themselves.

The algorithm works by looking ahead several moves. When it is your turn, you want to maximize your advantage, so you look for the move that leads to the best outcome. When it is the opponent's turn in the simulation, they want to minimize your advantage, so they choose the move worst for you. By alternating between maximizing and minimizing as we go deeper into the tree, we can evaluate how good each immediate move is by considering its long-term consequences.

At the bottom of the tree, when we have looked ahead as far as we want, we use an evaluation function to estimate how good the position is. This function considers factors like the score difference between players and potential future scoring opportunities. The minimax algorithm then propagates these evaluations back up the tree to determine the best move at the root.

#### Alpha-Beta Pruning
The problem with basic minimax is that the number of positions to evaluate grows exponentially with depth. If there are thirty possible moves at each position and we look ten moves ahead, we would need to evaluate thirty to the power of ten positions, which is far too many.

Alpha-beta pruning dramatically reduces this number by recognizing that many branches do not need to be explored. The key insight is that if we have already found a good move, and we discover that another move allows the opponent to force a worse outcome, we do not need to see exactly how much worse it could get. We can immediately stop exploring that branch and move on.

The algorithm maintains two values called alpha and beta. Alpha represents the best score the maximizing player can guarantee so far. Beta represents the best score the minimizing player can guarantee. Whenever beta becomes less than or equal to alpha, we know the current branch cannot possibly be chosen, so we prune it.

With good move ordering, alpha-beta pruning can reduce the effective branching factor from b to approximately the square root of b. This means instead of evaluating b to the power of d positions, we might only need to evaluate b to the power of d divided by two positions, which is a massive improvement.

#### Iterative Deepening
Iterative deepening addresses the time constraint problem. Instead of trying to search to a fixed depth and hoping we finish in time, we search to depth one first, then depth two, then depth three, and so on. After completing each depth, we have a best move to propose.

This approach might seem wasteful because we redo the shallow searches multiple times. However, the total work is dominated by the deepest search, so the overhead is acceptable. The crucial benefit is that we always have a move ready. If we get killed after completing depth five, we have the best depth-five move. If we get more time and complete depth six, we have an even better move.

We implement this by immediately proposing any legal move as soon as computation begins. This is our safety net. Then we propose the best greedy move, which considers only immediate scoring. Then we start iterative deepening, proposing increasingly better moves as we complete each depth level. The framework always uses the last move we proposed before being killed.

#### Move Ordering
The effectiveness of alpha-beta pruning depends heavily on the order in which we examine moves. If we happen to look at the best move first, we get maximum pruning. If we look at the worst move first, we get almost no pruning and alpha-beta degenerates to plain minimax.

Our AI orders moves by several criteria. First, if we have a best move stored from a previous search of this position, we try that move first because it is likely still good. Second, we prioritize moves that score points immediately because these are often strong moves. Third, we prefer moves in regions that are close to completion because these positions tend to offer scoring opportunities.

This ordering is heuristic, meaning it does not guarantee optimal ordering, but it significantly improves average performance by making good moves more likely to be examined early.

#### Transposition Tables
A transposition table is a cache that stores positions we have already evaluated. The name comes from chess, where the same position can be reached by different move orders, called transpositions.

In Sudoku, if player one plays cell A then cell B, and player two plays cell C, we reach a certain position. But if player one played cell B then cell A instead, we would reach the exact same position. Without a transposition table, our AI would evaluate this position twice, wasting time.

The transposition table stores each position along with its evaluation, the depth of search used, and the best move found. When we encounter a position, we first check if it is in the table. If we have already evaluated it to sufficient depth, we can reuse that result immediately. Even if the stored depth is insufficient, the stored best move helps with move ordering.

We must be careful about what we store. Because alpha-beta pruning can cut off search early, the stored value might be an upper bound, a lower bound, or an exact value. We use flags to track this, and we only reuse stored values when they are appropriate for our current alpha-beta window.


### The Evaluation Function
When the search reaches its depth limit or finds no legal moves, we need to estimate how good the position is. This is the job of the evaluation function. A good evaluation function is crucial because it guides the search toward favorable positions.

Our evaluation considers three factors. The most important is the score difference between us and the opponent, weighted heavily. This directly measures who is winning. Second, we consider region potential, giving small bonuses for almost-complete rows, columns, and blocks. Positions with many almost-complete regions offer more scoring opportunities. Third, we give a tiny bonus for mobility, meaning having more legal moves available. This captures the idea that having options is generally good.

The evaluation function must be fast because it is called many times during search. We avoid expensive computations and keep the logic simple while still capturing the essential aspects of position quality.


### Move Generation and Constraint Checking
Our AI generates legal moves by examining each empty square the current player is allowed to play on. For each square, we determine which values are forbidden by the Sudoku constraints. Any value already in the same row is forbidden. Any value already in the same column is forbidden. Any value already in the same block is forbidden.

The remaining values are candidates. We then filter out any moves that are in the taboo list. The moves that survive all these checks are legal and can be safely proposed. This constraint checking is critical because proposing a move that violates Sudoku rules causes immediate loss.


### Applying Moves During Search
During minimax search, we need to simulate making moves to see what positions they lead to. We do this by creating a copy of the game state and modifying the copy. We never modify the original state because the search explores many branches, and we need to backtrack by simply discarding the modified copy.
When applying a move, we calculate its score, update the player's total, place the value on the board, add the move to the history, update the occupied squares if applicable, and switch the current player. The resulting state represents the game after the move.


Putting It All Together
When the framework calls our AI to compute a move, the following sequence occurs. 
First, we clear the transposition table to start fresh. 
Second, we generate all legal moves for the current position. 
Third, we immediately propose the first legal move as a safety net.
Fourth, we order the moves by their immediate scores and propose the best greedy move. This ensures we have a reasonable move even if deeper search does not complete. Fifth, we enter the iterative deepening loop. For each depth, we search all moves using minimax with alpha-beta pruning. We use the transposition table to avoid redundant work and to improve move ordering. After completing each depth, we propose the best move found.

This loop continues until the framework kills our process. At that point, whatever move we most recently proposed becomes our official move.


### Why Our Approach Works
This combination of techniques creates a robust and effective AI. Minimax provides sound reasoning about opponent responses. Alpha-beta pruning makes deeper search feasible. Iterative deepening ensures we always have a move ready and naturally adapts to any time limit. Transposition tables avoid wasted computation. Move ordering maximizes the benefit of alpha-beta pruning. The evaluation function guides search toward winning positions.

Against opponents like the random player, which makes arbitrary legal moves, our AI wins easily because it thinks ahead and considers consequences. Against the greedy player, which only considers immediate scoring, our AI wins by seeing further into the future and setting up situations where it can score while denying scoring opportunities to the opponent.

The AI performs better as the second player on mid-game positions because it can react to the opponent's moves with full information about the current state. As the first player on empty boards, the search space is larger and less differentiated, making deep analysis harder.


### Conclusion
Building this Competitive Sudoku AI demonstrates how classical game-playing algorithms combine to solve complex problems. Each technique addresses a specific limitation. Minimax handles adversarial reasoning but is slow. Alpha-beta makes minimax practical but depends on move ordering. Iterative deepening handles time constraints gracefully. Transposition tables eliminate redundant work. Move ordering maximizes pruning efficiency. The evaluation function provides guidance when search must stop.

Understanding these techniques individually and how they interact is essential not just for this assignment but for any game-playing AI. The same principles apply to chess, checkers, Go, and countless other games. The specific details change, but the fundamental ideas of searching game trees, pruning unpromising branches, managing time, and evaluating positions remain constant across all adversarial game-playing AI systems.
