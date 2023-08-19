import Tribe from '/src/core/tribe';
import { predictVillages } from './prediction';

const DECIMAL = 10000;

// UNTESTED
// lvl 2 and 3 are worth more than lvl 4 and 5
// worksgop > stars > expand > unit
// ! bonus for each potential harvest?
// may prefer expand over pop?


// TODO multipliers
// opening
// middlegame
// endgame

// https://docs.google.com/document/u/0/d/1ZvDRejjXxgwr-NjIavDckoCeHrz4zUkagzpt9Xk0Nl8/mobilebasic

const weights: number[] = [
  [
    // ? penalty per turn + 1
    -0.7,
    // ? score per city level
    3.0,
    // ? score per pop / level
    0.5,
    // ? score per stars x turn
    2.0,
    // ? score per struct
    0.2,
    // ? score per tech * tier
    0.05,
    // ? worth() multiplier
    0.5,
    // ? bonus total 3 kills
    0.3,
    // ? bonus for terrain discovery.
    0.03,
    // ? bonus when in enemy territory.
    0.3,
    // ? bonus near enemy territory.
    0.15,
    // ? bonus for attacked foes.
    0.04,
    // ? bonus for defended allies.
    0.02,
    // ? bonus for getting close to a village / city.
    0.2, // v
    0.4, // c
    // ? bonus when sieging cities / villages.
    0.2, // v
    0.4, // c
    -0.5,
    1.0
  ],
  [
    -0.4,
    3.5,
    0.5,
    1.0,
    0.2,
    0.05,
    0.5,
    0.1,
    0.05,
    0.30,
    0.15,
    0.04,
    0.06,
    0.08,
    0.12,
    0.4,
    0.25,
    -0.5,
    3.0
  ],
  [
    -0.4,
    3.0,
    0.2,
    0.6,
    0.05,
    0.5,
    0.1,
    0.001,
    0.30,
    0.15,
    0.04,
    0.06,
    0.08,
    0.12,
    0.4,
    0.25,
    -0.5,
    4.0
  ]
];

const tableEcoCache: { string: number } = {};

function cacheEco(key: string, value: number | number[]): void {
  tableEcoCache[key] = Array.isArray(value) ? value : [value];
}

function cacheScore(id, score): string {
  return uid + [
      ...Object.keys(cache).sort().map(k => state[k].join('_'))
    ].join('-');
}

function serializeCache(cache): string {
  return uid + [
      ...Object.keys(cache).sort().map(k => state[k].join('_'))
    ].join('-');
}

export default function Score(tribe: Tribe, _: number[]): number {
  const state = cacheState('e', tableEcoCache);

  // 1 - 5 / 5 - 12 / 12 - 
  const stage = tribe._turn > 4 ? tribe._turn > 11 ? 2 : 1 : 0;

  const [SxEa, SxEb, SxEc, SxEd, SxEe, SxEf,
  SxUf, SxUg, SxUa, SxUb, SxUc, SxUd, SxUe, SxUf, SxUg, SxUh, SxUi, SxUj, SxUk] = weights[stage];

  let score = tableEcoCache[state] || 0;

  if (!score)
  {
    // TODO Caching must be put back in the tribe?
    // ?! goCache(uid, key, value);
    // ?! getCache(uid);
    
    score += SxEa * (tribe._turn + 1);

    const spt = tribe.getSpt();

    // ?! CACHE SPT
    cacheEco('sxed', spt);

    score += SxEd * spt;

    // ?! CACHE CITIES
    cacheEco(
      'sxec',
      tribe.getCities().map(c => {
        const lvl = c.getLvl(tribe); // 2 = 3 pop
        const pop = c.getPop(tribe); // 1/(lvl+1)
        // ? Anything over lvl 4 is not recommended
        const t = ((lvl > 4 ? 4 : lvl) / 4);
        // ? Bonus city level
        score += t * t * SxEb;
        // ? Small penalty for incomplete progress
        const half = (lvl + 1) * .5;
        score += pop ? ((pop - half) / half) * SxEc : 0;
        return c.getCache();
      })
    );
    
    // ?! CACHE TECH
    cacheEco(
      'tech',
      tribe.getTech().map(t => {
        score += SxEf * t.getTier();
        return t.getTier();
      })
    );
   

    tribe.getStructures().forEach(c => {
      score += SxEe;
      
      // ? mine | ore
      const adj = c._struct?.adjacent;
      if (adj)
      {
        // only good if 3+ adj
        const x = c.getAdjTerritory(tribe, c => adj.is(c._struct)).length;

        // if any adj resource is potential struct
        const y = c.getAdjTerritory(tribe, c => adj.required?.is(c.getResource(tribe))).length;

        // max 8 adjacent? but insaely rare
        // 5 top, 3 base for good
        score += ((x + y) - 1.5) * SxEe - y * SxEe * .7;
      }
      return [];
    });

    tableEcoCache[state] = score;
  }
  /* unit position score
   ★ Tiles attack / protected.
   ★ Tiles available for escape.
   ★ If unit is standing on enemy resource
   ★ The further away a unit is from the enemy the less its worth, or in other words, the less active it is?

   ★ (0.08) Bonus when in unclaimed territory.
   Bonusnfor protected squqres that enemy can move to
   ★ (0.008) Bonus for available steps.
  */

  const army = tribe.getArmy();
  const { terrain } = tribe.map;

  // Penalize for moves that are protected by foes
  // Penalize if adj foes can kill this unit

  const s = tribe.map.size;

  const scoreUnit = (u, vils) => {
    const cell = terrain[u._ix];
    const t = u.tribe;
    let _s = SxUf * u.worth();

    // ? Push for stacking kills
    _s += SxUg * ((u._kills > 3 ? 3 : u._kills) / 3) * u.getVitality();

    // ? Unexplored tiles
    const uL = cell.getAdj(terrain, 'a').filter(c => !t.isExplored(c)).length;

    _s += SxUa * uL;

    // ! THREAT ASSESSMENT

    // !? Bonus when in foe land and sieging same city
    // !? Bonus when defending high-risk units

    // ? Bonus when in foe territory
    // (only when has lot of health)
    const hpPenalty = u.getVitality() - .45;

    if (cell.isTerritory() && !cell.isMine(t))
    {
      _s += SxUb * hpPenalty;
    }
    // ? Bonus near foe border
    else if (cell.getAdj(terrain).some(c => c.isTerritory() && !c.isMine(t)))
    {
      _s += SxUc * hpPenalty;
    }

    // ! SUPPORT AND SYNERGY

    const adj = cell.getAdj(terrain, u.range);
    const iR = u.range > 1 ? 1.7 : 1;

    // ? Make ranged units stay away from nearby foes
    if (iR > 1)
    {
      _s += SxUj * cell.getAdj(terrain).filter(c => c._unit?.isFoe(t)).length;
    }

    // ? Bonus for attacked foes
    _s += SxUd * adj.filter(c => c._unit?.isFoe(t)).length * hpPenalty * iR;

    // ? Bonus for assisted allies
    // !? using ** causes weirdness
    _s += SxUe * adj.filter(c => c._unit && !c._unit.isFoe(t)).length * iR;

    vils.forEach(c => {
      const x = u._ix % s,
        y = Math.floor(u._ix / s);
      const xx = c.ix % s,
        yy = Math.floor(c.ix / s);

      let dist = c.is(u._ix) ? 0 : Math.max(Math.abs(xx - x), Math.abs(yy - y));
      const r = 3;
      const d = dist;

      dist = (r - (dist > r ? r : dist)) / r;

      // ? Push towards cities / villages
      _s += dist * SxUf;

      if (t.isExplored(c))
      {
        const iF = c._city?.isFoe(t) ? SxUk : 0;

        // ? Bonus if city is foe
        _s += dist * (c.isCity() ? SxUg - SxUf + iF : 0);
        // ? Is sieging?
        _s += (!d ? 1 : 0) * (c.isCity() ? SxUh + iF : SxUi);
      }
    });

    return _s;
  };

  const vils = predictVillages(tribe);

  army.forEach(u => {
    score += scoreUnit(u, vils);
  });

  tribe._memArmy.forEach(u => {
    // ? find our cities from their pov
    score -= scoreUnit(u, u.tribe.getExplored().filter(c => c.isVillage() || (c.isCity() && u.isFoe(c.city.tribe))));
  });

  tribe.getExplored().forEach(c => {
    const u = c._unit;
    if (!u || !u.isFoe(tribe)) return;
    score -= scoreUnit(u, u.tribe.getExplored().filter(c => c.isVillage() || (c.isCity() && u.isFoe(c.city.tribe))));
  });

  // TODO promote higher score moves by degrading moves that take longer (after a turn)

  // moved up ^
  //score += SxTu * (tribe._turn + 1);

  // TODO
  // penalozd for high ttr?
  // ttr cant go below zero except for free stars

  return Math.round(score * DECIMAL) / DECIMAL;
}