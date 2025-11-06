"""
move_search.py

Minimax-based move selection with alpha–beta pruning.
"""

import chess
from typing import Optional, List
from src.engine.evaluate import evaluate_board, move_value, check_end_game

# Mate scoring conventions:
# Use very large positive/negative scores to represent checkmates.
# MATE_THRESHOLD is used to detect mate-like scores and bias toward the quickest mate
# (by slightly reducing/increasing the score at deeper plies).
MATE_SCORE = 1000000000
MATE_THRESHOLD = 999000000


def minimax_move(depth: int, board: chess.Board) -> chess.Move:
    """
    Returns the next best move.
    """
    return minimax_root(depth, board)


def get_ordered_moves(board: chess.Board) -> List[chess.Move]:
    """
    Get legal moves.
    Attempt to sort moves by best to worst.
    Use piece values (and positional gains/losses) to weight captures.
    """
    end_game = check_end_game(board)

    def orderer(move):
        return move_value(board, move, end_game)

    in_order = sorted(
        board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
    )
    return list(in_order)


def minimax_root(depth: int, board: chess.Board) -> chess.Move:
    """
    Returns the highest value move per our evaluation function.
    """
    # White always wants to maximize (and black to minimize)
    # the board score according to evaluate_board()
    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)
        # Checking if draw can be claimed at this level, because the threefold repetition check
        # can be expensive. This should help the bot avoid a draw if it's not favorable
        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found


def _evaluate_leaf(depth: int, board: chess.Board, is_maximising_player: bool) -> Optional[float]:
    """
    Return a terminal/static value if this node is a leaf; otherwise None.
    - Checkmate: large +/- MATE_SCORE from the perspective of the side to move.
    - Other game over (stalemate/insufficient material): 0.
    - Depth == 0: static evaluation.
    """
    if board.is_checkmate():
        # Previous move delivered mate; from this side's POV it's losing/winning.
        return -MATE_SCORE if is_maximising_player else MATE_SCORE
    if board.is_game_over():
        return 0.0
    if depth == 0:
        return float(evaluate_board(board))
    return None


def _adjust_mate_distance(score: float) -> float:
    """
    Prefer faster mates / avoid slower mates by nudging very large scores.
    """
    if score > MATE_THRESHOLD:
        return score - 1
    if score < -MATE_THRESHOLD:
        return score + 1
    return score


def _search_child(depth: int, board: chess.Board, alpha: float, beta: float, is_max: bool, move: chess.Move) -> float:
    """
    Push -> recurse -> mate-distance adjust -> pop, returning the child's score.
    """
    board.push(move)
    try:
        child = minimax(depth - 1, board, alpha, beta, not is_max)
        return _adjust_mate_distance(child)
    finally:
        board.pop()


def minimax(depth: int, board: chess.Board, alpha: float, beta: float, is_maximising_player: bool) -> float:
    """
    - Early-exit on leaf nodes (_evaluate_leaf).
    - Single loop for both maximizing and minimizing nodes.
    """
    leaf = _evaluate_leaf(depth, board, is_maximising_player)
    if leaf is not None:
        return leaf

    moves = get_ordered_moves(board)

    best = -float("inf") if is_maximising_player else float("inf")
    for move in moves:
        curr = _search_child(depth, board, alpha, beta, is_maximising_player, move)

        if is_maximising_player:
            if curr > best:
                best = curr
            alpha = max(alpha, best)
        else:
            if curr < best:
                best = curr
            beta = min(beta, best)

        if beta <= alpha:
            break

    return best
