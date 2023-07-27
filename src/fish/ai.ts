import Tribe from '../core/tribe';
import Score from './score';
import { genUnitMoves, genEconomy, genArmy, genHires } from './generator';

function nextMoves(tribe, depth, moves, cb)
{
  let [_score, _ttr, _moves] = [Score(tribe), t2r(tribe), []];

  if (depth < 1)
  {
    return [_score, _ttr, _moves];
  }

  // ? mobile optimization
  moves = getBest(tribe, (moves ? moves : cb(tribe)), 6, _score);

  !moves.some(m => m.type === 'skip') && moves.push(new Pass())

  moves.forEach((m, i) => {
    count++;

    if (!m.play(tribe)) return;

    const _ss = tribe._stars;

    const [sc, ttr, ms] = nextMoves(
      tribe,
      depth - 1,
      m.type === 'skip' ? null : moves.filter((_, x) => i !== x),
      cb
    );

    m.undo();

    if (sc < _score)
    {
      return;
    }

    _score = sc;
    _ttr = ttr;
    _moves = [
      [m._score, m._ttr, m, _ss, m.cell?.ix],
      ...m._rewards,
      ...ms
    ];
  });

  return [_score, _ttr, _moves];
}

// Define a function to generate all possible moves from the current game state
function generateMoves(gameState) {
  // Implementation to generate possible moves based on the current game state
}

// Define a function to evaluate the current game state (Heuristic function)
function evaluate(gameState) {
  // Implementation to evaluate the game state and return a numeric score
}

function alphaBetaSearch(gameState, depth, alpha, beta, maximizingPlayer) {
  if (depth === 0) {
    return [evaluate(gameState), []];
  }

  if (maximizingPlayer) {
    let value = -Infinity;
    let bestMoves = [];

    const moves = generateMoves(gameState);
    for (const move of moves) {
      const newGameState = 0/* apply move to get the new game state */ ;
      const [newValue, newMoves] = alphaBetaSearch(newGameState, depth - 1, alpha, beta, false);

      if (newValue > value) {
        value = newValue;
        bestMoves = [move, ...newMoves];
      }

      alpha = Math.max(alpha, value);
      if (beta <= alpha) {
        break; // Beta cut-off
      }
    }
    return [value, bestMoves];
  } else {
    let value = Infinity;
    let bestMoves = [];

    const moves = generateMoves(gameState);
    for (const move of moves) {
      const newGameState = 0/* apply move to get the new game state */ ;
      const [newValue, newMoves] = alphaBetaSearch(newGameState, depth - 1, alpha, beta, true);

      if (newValue < value) {
        value = newValue;
        bestMoves = [move, ...newMoves];
      }

      beta = Math.min(beta, value);
      if (beta <= alpha) {
        break; // Alpha cut-off
      }
    }
    return [value, bestMoves];
  }
}

// Function to find the best list of moves using alpha-beta search
function findBestMoves(gameState, depth) {
  const moves = generateMoves(gameState);
  let bestMoves = [];
  let bestValue = -Infinity;
  const alpha = -Infinity;
  const beta = Infinity;
  const maximizingPlayer = true;

  for (const move of moves) {
    const newGameState = 0/* apply move to get the new game state */ ;
    const [value, newMoves] = alphaBetaSearch(newGameState, depth - 1, alpha, beta, !maximizingPlayer);

    if (value > bestValue) {
      bestValue = value;
      bestMoves = [move, ...newMoves];
    }
  }

  return bestMoves;
}