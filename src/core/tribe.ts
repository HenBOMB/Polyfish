import Unit from './unit';
import Cell from './cell';
import City from './city';
import Tech, { Tree } from '../tech/tech';
import Resource from '../tech/resource';
import Structure from '../tech/structure';

export default class Tribe
{
  id: string;
  terrain: Cell[];
  foes: Tribe[] = [];
  cache: { string: { string: number[] } } = {};
  
  // ? sometimes after loosing a city, it will not get rmoved from the list
  // ? this is to keep track of winning the city back in the future or smth idk..
  private _cities: City[] = [];

  // ! tmp
  _turn: number = 0
  _score: number = 0;
  _stars: number = 5;
  _spent: number = 0;
  private _explored: Cell[] = [];
  private _tech: Tech[];
  private _truetech: string[];

  _memArmy: Unit[Unit[]] = [];

  constructor(id: string, tech: Tech)
  {
    this.id = id;
    this._tech = [tech];
    this._truetech = [tech];
    this.addScore(100 * tech.getTier());
  }
  
  setCache(uid: string, key: string, value: number | number[]): void {
    
  }
  
  getCache(uid: string, key: string): number[] {
    const cache = this.cache[uid] && this.cache[uid][key] || [];
    return cache;
  }
  
  packCache(uid): string {
    const cache = this.cache[uid] || [];
    return uid + Object.keys(cache).sort().map(k => cache[k].join('_')).join('-');
  }
  
  doMoveUnit(from: Cell, to: Cell)
  {
    if (!from._unit || to._unit)
    {
      return false;
    }

    to._unit = from._unit;
    from._unit = null;

    return () => {
      from._unit = to._unit;
      to._unit = null;
    };
  }

  // ?! do's dont return undo

  doEnable()
  {
    this.getArmy().map(u => u.goReplenish());
  }

  doDiscoverFoe(foe: Tribe)
  {
    // ? 2*min( ceil(score/1000) ,5)+1
    const stars = 2 * Math.min(Math.ceil(foe.getScore() / 1000, 5)) + 1;
    this.goBuy(-stars);
    this.foes.push(foe);
  }

  doScore(score: number)
  {
    this._score = score;
  }

  is(other: Tribe | string)
  {
    return this.id === (other?.id || other);
  }

  isFoe(other: Tribe | null)
  {
    return !this.is(other);
  }

  isExplored(cell: Cell | number)
  {
    return this._explored.some(c => c.is(cell));
  }

  isBuyable(item)
  {
    return this._stars >= item.getCost(this);
  }

  hasTech(tech: Tech | string, abs: boolean)
  {
    return [this._truetech, this._tech][abs ? 0 : 1].some(t => t.is(tech));
  }

  getCache(key: string): any[] {
    return this._cacheEco[key] ? this._cacheEco[key] : [];
  }

  // returns all unowned but unlocked tech
  getUnlockedTech()
  {
    return Tree.filter((t) => t.isUnlocked(this));
  }

  // ? returns all owned tech
  getTech()
  {
    return this._tech;
  }

  // ? returns all cities ruled by this tribe
  getCities()
  {
    return this._cities.filter(c => c._tribe.is(this));
  }

  // ? returns cities under siege by this tribe
  getSiege()
  {
    const cities = this._explored.filter(c => c.city?.isSieged() &&
      !this.is(c.city._tribe) &&
      this.getArmy().some(u => u._ix === c.ix)
    ).map(c => c.city);
    return cities;
  }

  // ? returns cities with unit getting ready to siege
  getSiegable()
  {
    const cities = this._explored.filter(c =>
      c.city &&
      !c.city.isSieged() &&
      !this.is(c.city._tribe) &&
      this.getArmy().some(u => u._ix === c.ix)
    ).map(c => c.city);
    return cities;
  }

  // ? returns tribe capital city
  getCapital()
  {
    return this._cities[0];
  }

  getExplored()
  {
    return this._explored;
  }

  // ? returns territory occupied by cities
  getTerritory()
  {
    const cities = this.getCities();
    if (!cities.length) return null;
    return cities.map(c => c.getTerritory(this)).reduce((p, c) => [...p, ...c]);
  }

  // ? returns all cells with a structure in terrirory
  getStructures()
  {
    return this.getTerritory().filter(c => c.getStruct());
  }

  // ? returns all the tribe units
  getArmy()
  {
    if (!this._cities.length) return [];
    return this._cities.map(c => c.getArmy(this)).reduce((p, c) => [...p, ...c]);
  }

  // ? returns spt produced by cities / houses
  getSpt(): number {
    const cities = this.getCities();
    if (!cities.length) return 0;
    return cities.map(c => c.getSpt(this)).reduce((p, c) => p + c);
  }

  getTtr(): number {
    const t = this._spent / this.getSpt();
    return Math.round(t * 1000) / 1000;
  }

  getTotCost(tech: Tech)
  {
    let cost = tech.getCost(this);

    tech.getPrevious().forEach(t => {
      cost += t.getCost(this);
    });

    return cost;
  }

  getScore()
  {
    return this._score;
  }

  addScore(score: number, tag: string)
  {
    const tribe = this;
    tribe._score += score;
    return () => {
      tribe._score -= score;
    }
  }

  // ACTIONS

  // private
  goBuy(item: 'Buyable')
  {
    const tribe = this;
    const cost = Number.isInteger(item) ? item : item.getCost(tribe, true);

    if (tribe._stars < cost)
    {
      console.error('negative stars');
      console.trace();
    }

    tribe._stars -= cost;
    tribe._spent += cost;

    return () => {
      tribe._stars += cost;
      tribe._spent -= cost;
    };
  }

  goExplore(cell: Cell, terrain: Cell[], radius: number = 1)
  {
    const tribe = this;

    const uncharted = [cell, ...cell.getAdj(terrain, radius)];

    const explored = uncharted.filter(u => !tribe._explored.some(e => u.is(e)));

    const l = tribe._explored.length;
    tribe._explored.push(...explored);

    const sUndo = tribe.addScore(5 * explored.length);

    return () => {
      tribe._explored = tribe._explored.slice(0, l - 1);
      sUndo();
    };
  }

  goResearch(tech: Tech | null)
  {
    if (!tech || this.hasTech(tech))
    {
      return false;
    }

    const cost = tech.getCost(this);

    if (!tech.isUnlocked(this) || this._stars < cost)
    {
      return false;
    }

    const tribe = this;

    this._tech = [...this._tech, tech];

    const bUndo = this.goBuy(tech);
    const sUndo = this.addScore(100 * tech.getTier())

    const undo = () => {
      tribe._tech.pop();
      bUndo();
      sUndo();
    };

    return undo;
  }

  goCast(cell: Cell, ability)
  {
    if (!ability.canCast(cell))
    {
      return false;
    }

    return [ability.cast(cell), null];
  }

  goHarvest(cell: Cell)
  {
    const res = cell.getResource(this);

    if (!res || (res.required && !this.hasTech(res.required)))
    {
      return false;
    }

    if (cell._struct)
    {
      // TODO fatal
      return false;
    }

    if (this._stars < (res.pop > 0 ? res.cost : -1))
    {
      return false;
    }

    const cost = res.cost * (res.pop > 0 ? -1 : 1);

    cell.setResource(null);
    cell._struct = res.struct;

    const bUndo = this.goBuy(res);

    const reward = res.pop > 0 ? cell._belongsTo.goPopulate(res.pop) : null;

    const tribe = this;

    const undo = () => {
      bUndo();
      cell.setResource(res);
      cell.setStructure(null);
    };

    return [
      undo,
      reward,
    ];
  }

  goBuild(cell: Cell, struct: Structure)
  {
    if (cell.city)
    {
      return false;
    }

    if (!struct.canBuild(cell, this))
    {
      return false;
    }

    const cost = struct.cost;

    if (this._stars < cost)
    {
      return false;
    }

    const rewards = cell.getAdjTerritory(this, (c) => struct.is(c._struct?.adjacent)).map(c => c._belongsTo.goPopulate(c._struct.pop));

    const mult = struct.adjacent ? cell.getAdjTerritory(this, (c) => struct.adjacent.is(c._struct)).length : 1;

    const res = cell.getResource();

    cell.setResource(null);
    cell.setStructure(struct);

    const bUndo = this.goBuy(struct);

    const reward = cell._belongsTo.goPopulate(struct.pop * mult);

    function undo() {
      bUndo();
      res && cell.setResource(res);
      cell.setStructure(null);
    };

    return [
      undo,
      reward,
      rewards
    ];
  }

  doPass()
  {
    this._truetech = [...this._tech];

    this._memArmy.forEach(u => {
      const c = this.map.terrain[u._ix];
      c._unit = null;
    });

    this._memArmy = [];

    this.goPass();
  }

  goPass()
  {
    const tribe = this;

    const bUndo = this.goBuy(-tribe.getSpt());
    const rUndo = tribe.getArmy().map(u => u.goReplenish());

    tribe._turn++;

    return () => {
      rUndo.forEach(u => u());
      bUndo();
      tribe._turn--;
    };
  }
}