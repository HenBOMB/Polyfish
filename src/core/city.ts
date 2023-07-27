import Unit from './unit';
import Tribe from './tribe';
import Cell from './cell';

export default class City
{
  isCapital: boolean;
  // ! obsolete
  ix: number;
  cell: Cell;
  
  _tribe: Tribe | null;
  private _pop: number;
  private _lvl: number;
  private _army: Unit[];
  private _territory: Cell[];
  private _rewards: string[];

  constructor(cell, isCapital = false)
  {
    this.cell = cell;
    this.ix = cell.ix;
    this._tribe = null;
    this.isCapital = isCapital;
    this._pop = 0;
    this._lvl = 1;
    this._army = [];
    this._territory = [];
    this._rewards = [];
  }
  
  is(other: City | number)
  {
    return this.cell.ix === ((other?.cell?.ix || other) + 0);
  }
  
  isCity()
  {
    return this._tribe? true : false;
  }
  
  isFoe(tribe)
  {
    return this._tribe?.isFoe(tribe);
  }
  
  isSieged()
  {
    return !this.tribe && this.cell._unit?.isFoe(this._tribe);
  }
  
  isHireable(unit: Unit)
  {
    if (this.cell._unit || this._army.length > this._lvl) return false;
    return this._tribe._stars >= unit.cost;
  }
  
  has(rew: string)
  {
    return this._rewards.some(r => r === rew);
  }

  getTerritory(tribe: Tribe | null = null)
  {
    return !tribe || tribe.is(this._tribe) ? this._territory : [];
  }

  getArmy(tribe: Tribe | null = null)
  {
    if (!tribe) return this._army;
    return this._army.filter(u => tribe.is(u.tribe));
  }

  getSpt(tribe: Tribe)
  {
    if (!this._tribe.is(tribe) || this.isSieged()) return 0;

    return this.getLvl(tribe) + (this.isCapital? 1 : 0) + (this.has('workshop') ? 1 : 0);
  }

  getPop(tribe: Tribe)
  {
    if (!this._tribe.is(tribe)) return 0;
    return this._pop - this._army.filter(u => !u.tribe.is(this._tribe)).length;
  }

  getLvl(tribe: Tribe)
  {
    if (!this._tribe.is(tribe)) return 0;

    return this._lvl - (this.getPop(tribe) < 0 ? 1 : 0)
  }

  private getRewards()
  {
    switch (this._lvl)
    {
      case 2:
        return ['workshop', 'explorer'];

      case 3:
        return ['stars', 'walls'];

      case 4:
        return ['border', 'growth'];

      default:
        return ['unit', 'park'];
    }
  }
  
  // ! ACTIONS
  
  goHire(unit: Unit)
  {
    if(!this.isHireable(unit))
    {
      return false;
    }
    
    const aUndo = this.goEnrole(unit);
    const sUndo = unit.goSpawn(this.cell);
    const bUndo = this._tribe.goBuy(unit);
    
    unit.tribe = this._tribe;
  
    return () => {
      aUndo();
      sUndo();
      bUndo();
      unit.tribe = null;
    }
  }
  
  goExpel(unit: Unit): ()=>void {
    const city = this;
    const home = unit._city;
    
    city._army = city._army.filter(u=>!u.is(unit));
    unit._city = null;
    
    return () => {
      city._army = [...city._army, unit];
      unit._city = home;
    };
  }
  
  goEnrole(unit: Unit): ()=>void {
    const city = this;
    const home = unit._city;
    //const eUndo = home?.goExpel(unit);
    
    if(home) home._army = home._army.filter(u=>!u.is(unit));
    city._army = [...city._army, unit];
    unit._city = city;
    
    return () => {
      city._army.pop();
      if(home) home._army = [...home._army, unit];
      unit._city = home;
    };
  }
  
  // ! [undo, labels, options];
  goPopulate(pop: number)
  {
    const pop_ = this._pop;
    const lvl_ = this._lvl;
    const city = this;
    const tribe = this._tribe;
    
    let rew = null;
    let score = 5 * pop;
    
    this._pop += pop;

    if (this._pop > this._lvl)
    {
      this._lvl++;
      this._pop -= this._lvl;
      rew = this.getRewards();
      const s = -this._lvl * 5 + 50;
      score += s<0?0:s;
    }
    
    const sUndo = tribe.addScore(score);
    
    const undo = () => {
      city._pop = pop_;
      city._lvl = lvl_;
      sUndo();
    };

    return [
      undo,
      rew,
      rew ? [
        () => city.applyReward(rew[0]),
        () => city.applyReward(rew[1])
      ] : null
    ];
  }

  goCapture()
  {
    const unit = this.cell._unit;
    
    if(!unit)
    {
      return false;
    }
    
    // TODO
    // other tribe still hss access to military
    // they have to discard it
    if (this._tribe)
    {
      //this._tribe._cities = this._tribe._cities.filter(c=>c.ix!==this.ix);
    }
    
    const terr = [...this._territory];
    let score = 0, sUndo = null;
    
    // TODO
    // city may overlap with other territories
    if (!this._tribe)
    {
      const adj = Cell.getAdj(unit._ix, 1, true);
      this._territory = unit.tribe._explored.filter(c=>adj.some(i=>i===c.ix));
      this._territory.forEach(c => c._belongsTo=this);
      sUndo = unit.tribe.addScore(20 * this._territory.length + 100);
    }
      
    const old = this._tribe;
    const city = this;
    
    const tUndo = city.goEnrole(unit);
    
    this._tribe = unit.tribe;
    this._tribe._cities = [...this._tribe._cities, this];
    
    this._tribe.addScore(score);
    
    const exUndo = unit.goExaust(true);
    
    return () => {
      tUndo();
      sUndo && sUndo();
      city._tribe._cities.pop();
      city._tribe.addScore(-score);
      city._tribe = old;
      terr.forEach(c => c._belongsTo = city);
      this._territory = terr;
      exUndo && exUndo();
    };
  }
  
  private applyReward(reward: string)
  {
    const city = this;
  
    switch (reward) {
      case 'workshop':
      case 'walls':
        city._rewards = [...city._rewards, reward];
        return () => city._rewards.pop();
  
      case 'explorer':
        return () => {};
  
      case 'stars':
        return city._tribe.goBuy(-5);
  
      case 'growth':
        city._pop += 4;
        return () => city._pop -= 4;
  
      case 'border':
        return () => {};
  
      case 'park':
        city._tribe.addScore(250, 'park');
        return () => city._tribe.addScore(-250);
  
      case 'unit':
        const giant = new Unit('giant', 5, 4, 40, 1, 1, 10, city);
  
        const score = giant.getScore();
  
        city._tribe.addScore(score);
  
        return () => {
          city._army.pop();
          city._tribe.addScore(-score);
        }
  
    }
  
    return null;
  }
}