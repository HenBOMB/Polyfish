import Cell from '../core/cell';
import Tribe from '../core/tribe';
import Tech from '../tech/tech';
import Move, { Pass } from './move';
import { genUnitMoves, genEconomy, genArmy, genHires } from './generator';
import Score from './score';

var count;

// TODO
// pass in game state
// move mem to state, pov
export default function evaluate(tribe: Tribe, depth = 5)
{
  count = 0;

  const eMoves = nextMoves(
    tribe,
    depth,
    (t) => {
    const army = t.getArmy().map(u => getBest(t, genUnitMoves(u), 3, 0)).reduce((a, b)   => [...a, ...b]);
      return [...army, ...genEconomy(t)];
    }
  );
  
  return [...eMoves, count];
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

function nextMoves(tribe, depth, cb, moves)
{
  // ?! Get cached score state
  let [_score, _ttr, _moves] = [Score(tribe), tribe.getTtr(), []];

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
      cb,
      m.type === 'skip' ? null : moves.filter((_, x) => i !== x)
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

// this is NOT negamx
function search(moves: Move[] | null, tribe: Tribe, depth: number, cbGen, cpScore = -999)
{
  // ? skip, most likely
  if (!moves)
  {
    moves = cbGen(tribe);
  }

  // future: sim other tribes turns
  if (depth < 1 || !moves.length)
  {
    //if (!moves.length)
    //{
    return [Score(tribe), tribe.getTtr(), []];
    //}

    // this causes extra nodes to be searched
    // leads to worse score
    //moves = moves.filter(m=>m.type!=='skip');
  }

  let [_score, _ttr, _moves] = [-999, 999, []];

  moves.forEach((m, i) => {
    if (!m.play(tribe)) return;

    const _s = m._score;
    const _t = m._ttr;
    const _ss = tribe._stars;

    count++;

    const nm = m.type === 'skip' ? null : moves.filter((_, x) => i !== x);

    let [score, ttr, ms] = search(
      nm,
      tribe,
      depth - 1,
      cbGen,
      cpScore > 0 && !nm ? _s : cpScore);

    // TEST
    const penalize = (p) => {
      score -= p;
      _s -= p;
    };

    // ? if less than no progress has been made
    if (_s - cpScore < 0 && m.type === 'skip')
    {
      //penalize(1);
      m.undo();
      return;
    }

    // ? exausr only when any other paths are worse
    // ? longer steps is better?
    // if wants to move to an ubdiscovered square, wait

    // penalty: -2 - 0
    /*
    const im = tribe.getCities().filter(c => c.getPop(tribe)).map(c => c.getPop(tribe) / (c.getLvl(tribe) + 1));
    im.forEach(v => {
      const p = !v ? 0 : (v * .5);
      // ? problem: farm goves .1 points
      // ! so building in this case is bad
      penalize(p);
    });
    */
    // uncaptured siege

    //same score with higher score is better

    // pass then siege cannot happen

    if (_score >= score)
    {
      if (score === -999)
      {
        _ttr = _t;
        _score = _s;
      }
      else
      {
        m.undo();
        return;
      }
    }
    else
    {
      _ttr = ttr;
      _score = score;
    }

    _moves = [
      [_s, _t, m, _ss, m?.cell?.ix],
      ...m._rewards,
      ...ms
    ];

    m.undo();
  });

  return [_score, _ttr, _moves];
}
