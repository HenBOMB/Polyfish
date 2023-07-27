import Cell from '../core/cell';
import City from '../core/city';
import Unit from '../core/unit';
import { Fish, Game, Whale, Ore, Fruit, Crop } from '../tech/resource';
import Move, { Construct, Purchase, Harvest, Capture, Pass, Cast, Step, Attack, Heal, Exaust, Hire } from './move';

import { Warrior } from '/src/units';

// TODO ADD MISSING STUFF FROM HERE:
// ! https://docs.google.com/document/u/0/d/1I82qRpIx1ynklrE8J8rHGYiKBc-8jdbX71hyqhtOGAs/mobilebasic
// ! https://docs.google.com/document/u/0/d/1D5jMQVxpmyFtDlw-S_21uH1IsSWPY-4A7u9miuWeals/mobilebasic

export function genMap(map)
{
  map = map.map(m => m.match(/[\w_]+/g).join(''));

  const len = map[0].length;
  const terrain = Array(len);

  for (var i = 0; i < len; i++)
  {
    const ty = map[0][i];
    const ab = map[1][i];

    terrain[i] = new Cell((() => {
      switch (ty)
      {
        case 'F':
          return 'forest';
        case 'M':
          return 'mountain';
        case 'W':
          return 'water';
        case 'O':
          return 'ocean';
        case 'C':
          return 'capital';
        default:
          return 'field';
      }
    })());

    const cell = terrain[i];

    switch (ab)
    {
      case 'G':
        cell.setResource(Game);
        break;
      case 'F':
        cell.setResource(Fruit);
        break;
      case 'S':
        cell.setResource(Fish);
        break;
      case 'E':
        cell.setResource(Whale);
        break;
      case 'O':
        cell.setResource(Ore);
        break;
      case 'C':
        cell.setResource(Crop);
        break;
      case 'V':
        cell.city = new City(cell, ty === 'C');
        break;
      case 'R':
        // TODO ruins
        // cell._resource = null;
        break;
    }
  }
  
  terrain.forEach(c => {
    c.cacheAdj[1] = Cell.getAdj(c.ix, 1).map(i => terrain[i]);
    c.cacheAdj[2] = Cell.getAdj(c.ix, 2).map(i => terrain[i]);
  });
  
  return {
    terrain,
    len,
    size: Math.sqrt(len),
  };
}

export function genEconomy(tribe, strict=false): Move[] {
  const land = tribe.getTerritory();
  const harvests = [];
  const techs = [];
  const others = [];
  const builds = [];
  const abilities = [];

  let imp = 0;

  // checks if has id in list
  const has = (arr, match) => {
    return arr.some(x => x.id === match.id)
  };

  // TODO 
  // non included harvests may not be subject to, may miss a good move. +5 stars
  land.forEach(c => {
    const r = c.getResource(tribe);

    if (!r) return;

    if (c.struct) return;
  
    if (!tribe.isBuyable(r))
    {
      if (strict) return;
      imp++;
    }
    
    // TODO not compatible with ruins
    harvests.push(new Harvest(r.id, c));

    if (!tribe.hasTech(r.required))
    {
      techs.push(new Purchase(r.required));
    }
  });

  const terr = tribe.getTerritory();

  const unlocked = tribe.getTech().filter(t => t.unlocks.struct).map(t => t.unlocks);

  const tech = tribe.getUnlockedTech();
  tech.forEach(t => {
    // ? if req for harvest?
    
    if(strict && !tribe.isBuyable(t)) return;
    
    // if tech struct can be used
    if (t.unlocks.struct && terr.some(c => t.unlocks.struct.canBuild(c, tribe)))
    {
      techs.push(new Purchase(t));
      return;
    }

    // if tech struct has an adj struct we already own
    const adj = t.unlocks.struct?.adjacent;
    if (adj && unlocked.some(u => adj.is(u.struct)))
    {
      techs.push(new Purchase(t));
      return;
    }

    // if tech ability can be used

    //techs.push(new Purchase(t));
  });

  terr.forEach(c => {
    unlocked.forEach(t => {
      if (!t.struct || (strict && !tribe.isBuyable(t.struct))) return;
      if (t.struct.canBuild(c, tribe)) builds.push(new Construct(t.struct, c));
      // TODO 
      //if (t.ability?.canCast(c)) abilities.push(new Cast(t.ability, c));
    });
  });

  //const siege = tribe.getSiege();

  //others.push(...siege.map(c => new Capture(c)));

  // avoid unnescessary passes?
  //if ((builds.length + harvests.length) < 3 /* || tribe._stars < tribe.getSpt() * 1.6*/ );
  {
  harvests.push(new Pass());
  }

  return [...techs, ...others, ...abilities, ...builds, ...harvests];
}

export function genArmy(tribe): Move[] {
  const army = tribe.getArmy();
  let moves = [];

  army.forEach(u => {
    moves = [...genUnitMoves(u), ...moves];
  });
  
  return [...moves, new Pass()];
}

export function genHires(tribe): Move[] {
  let moves = [];
  
  const cities = tribe.getCities();
  const fn = (c, u) => {
    //if (!c.isHireable(u)) return;
    moves = [...moves, new Hire(u, c)];
  };
 
  cities.forEach(c => fn(c, Warrior()));
  
  tribe.getTech().filter(t => t.unlocks.unit).forEach(t => {
    cities.forEach(c => fn(c, t.unlocks.unit()));
  });
  
  return moves;
}

export function genUnitMoves(unit: Unit): Move[] {
  // ! TODO limited to range of 1
  const tribe = unit.tribe;

  const atk = unit.getCell().getAdj(tribe.map.terrain, 1, (c) => c._unit && !c._unit.tribe.is(tribe), unit.range).map(c => new Attack(c._unit, c));

  const mov = unit.getCell().getAdj(tribe.map.terrain, 1, (c) => unit.isSteppable(c)).map(c => new Step(unit, c));

  const actions = genActions(unit);
  
  return [...actions, ...mov, ...atk, new Exaust(unit)];
}

function genActions(unit: Unit): Move[] {
  const moves = [];

  const city = unit.tribe.getSiege().find(c => c.is(unit._ix));

  if (city)
  {
    moves.push(new Capture(city));
  }

  return [new Heal(unit), ...moves];
}