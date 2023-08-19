import Tribe from '../core/tribe';
import Score from './score';
import { genUnitMoves, genEconomy, genArmy, genHires } from './generator';

// TODO
/*
Doubt this will ever work..
(incomplete)
*/
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

function getBest(tribe, moves, range, score): any[] {
  return moves.sort((a, b) => {
    const s1 = 0,
      s2 = 0;

    if (a.play(tribe))
    {
      s1 = Score(tribe) - score;
      a.undo();
    }

    if (b.play(tribe))
    {
      s2 = Score(tribe) - score;
      b.undo();
    }

    return s2 - s1;
  }).slice(0, range);
}

var gen = null;
var count = 0;

// Define a function to generate all possible moves from the current game state
function generateMoves(tribe, cb = gen) {
  if (cb && !gen) gen = cb;
  return getBest(tribe, cb? cb : gen, 4, -999);
}

function alphaBetaSearch(tribe, depth, alpha, beta, maximizingPlayer): void {
  
  if (depth === 0) return [Score(tribe), []];

  if (maximizingPlayer) 
  {
    let score = -Infinity;
    let bestMoves = [];

    const moves = generateMoves(tribe);
    
    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      
      if(!m.play(tribe)) continue;
      
      const [newScore, newMoves] = alphaBetaSearch(tribe, depth - 1, alpha, beta, false);
      
      m.undo();

      if (newScore > score) {
        score = newScore;
        
        bestMoves = [m._format, ...m._rewards, ...newMoves];
      }

      alpha = Math.max(alpha, score);
      if (beta <= alpha) {
        break; // Beta cut-off
      }
    }
    return [score, bestMoves];
  } 
  else 
  {
    let score = Infinity;
    let bestMoves = [];

    const moves = generateMoves(tribe);
    for (const move of moves) {
      if(!move.play(tribe)) continue;
      
      const [newScore, newMoves] = alphaBetaSearch(tribe, depth - 1, alpha, beta, true);
      
      move.undo();
      
      if (newScore < score) {
        score = newScore;
        bestMoves = [move, ...newMoves];
      }

      beta = Math.min(beta, score);
      if (beta <= alpha) {
        break; // Alpha cut-off
      }
    }
    return [score, bestMoves];
  }
}

// Function to find the best list of moves using alpha-beta search
function findBestMoves(tribe, depth, gen) {
  const moves = generateMoves(tribe, gen);
  let bestMoves = [];
  let bestScore = -Infinity;
  const alpha = -Infinity;
  const beta = Infinity;
  const maximizingPlayer = true;

  for (const move of moves) {
    if(!move.play(tribe)) continue;
    
    const [score, newMoves] = alphaBetaSearch(tribe, depth - 1, alpha, beta, !maximizingPlayer);
    
    move.undo();
    
    if (score > bestScore) {
      bestScore = score;
      bestMoves = [move, ...newMoves];
    }
  }

  return bestMoves;
}