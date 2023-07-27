import { Vengir, Imperius, XinXi } from '/src/tribes';
import { Warrior } from '/src/units';
import Cell from '/src/core/cell';
import { Hunting } from '/src/tech/tech';
import { genMap } from '/src/fish/generator';
import evaluate, { evalArmy, t2r, DECIMAL } from '/src/fish/evaluate';
import Score from '/src/fish/score';
import { Farm } from '/src/tech/structure';

import Map from './maps/03';

Cell.len = 121;

const map = genMap(Map);

const { terrain } = map;

const capital = terrain.find(c => c.city?.isCapital)?.city;

const tribe = Imperius(capital, map);

// some start game logic 

tribe.doEnable();

const cell = i => terrain[i];

const claim = (rew, choice = 0) => {
  return rew[1][2][choice]();
};

const [ unit ] = tribe.getArmy();

//**//**//**//**//**//**//

unit.goMove(cell(80), [terrain]);
tribe.goHarvest(cell(78));
tribe.goHarvest(cell(100))[1][2][0]();
tribe.doPass();

capital.goHire(Warrior(capital));
unit.goMove(cell(70), [terrain]);
tribe.doPass();
   
// ?! after playing a true move that explores tiles, like a step, ai can re-calc to find better path

//unit.goMove(cell(71), [terrain]);

/*
// ! discovered xin-xi and unit
const xin = XinXi(null, map);
xin.goDiscover(48, map.terrain, 2);
const foe = Warrior(null);
foe.goReplenish();
foe.tribe = xin;
foe.goSpawn(cell(36));
foe.goMove(cell(48), [map.terrain]);
xin.doDiscoverFoe(tribe);
xin.doScore(900);
xin.doPass();

tribe.doDiscoverFoe(xin);
*/

//**//**//**//**//**//**//

/*
import MCTS from '/src/fish/mcts';
const mcts = new MCTS(tribe, 3, 2000);
*/

// !! PERFORMANCE TESTING
function benchmark(fn, iterations) {
  const startTime = Date.now();

  for (let i = 0; i < iterations; i++) {
    fn();
  }

  const took = Date.now() - startTime;
  
  console.log(`took: ${took/1000}s`);
  console.log(`cycle: ${took/iterations}ms`);
}

// ~3 times x cycle, 1 time x move
// took: 24964 ms
// cycle: 0.0554 ms
benchmark(() => Score(tribe), (25000*3*6)/4);
return
// took: 14934.099999904633 ms
// searched 18648 nodes

x: {
  //break x;
  console.time('took');

  //const [score, ttr, moves, states, count] = mcts.search();
  const ms = [];//evaluate(tribe, 7)[2];
  const undos = ms.map(([,,m])=>m.play&&m.play(tribe));
  
  let [score, ttr, moves, count] = evaluate(tribe, 6);
  
  moves = [...ms, ...moves];
  
  console.timeEnd('took');
  console.log(`searched ${count} nodes`);

  const _s = Score(tribe);
  const _t = t2r(tribe);
  const _ss = tribe._stars;
  
  const st = (l, a) => {
    return `${l}: ${a}`;
  }
  
  const uh = (l, x, _x, i, _i) => {
    const v = i ? moves[i - 1][_i] : _x;
    if (!(x - v)) return null;
    return `${l}: ${moves[i][_i]} (${x>v?'+':'-'}${Math.round(Math.abs(x-v)*DECIMAL)/DECIMAL})`;
  }
  
  moves.forEach(([s, t, m, ss, ix], i) => {
    let strs = [];
    
    strs.push(...((m.custom && m.custom())||[`${m.id} (${m.type})`]));
    
    let s_ = uh('score', s, _s, i, 0);
    if (s_) strs.push(s_);
    else strs.push('score: '+s);
    
    strs.push(uh('ttr', t, _t, i, 1));
    strs.push(uh('stars', ss, _ss, i, 3));
    
    if (ix) strs.push(`ix: ${ix}`);
    
    /*
    const state = states[i];
    strs.push(st('turn', state.tribe._turn));
    
    if (state.action.choices.length)
      strs.push(`reward: ${state.action.choices[0]}`);
    */
    
    strs = strs.filter(Boolean).map(s => `${s}\n`);
    
    console.log(...strs);
  });
}