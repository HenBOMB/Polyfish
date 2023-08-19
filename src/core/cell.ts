import Tribe from './tribe';
import City from './city';
import Unit from './unit';

import { Fishing, Organization, Climbing } from '../tech/tech';
import Structure from '../tech/structure';
import Resource, { Fish, Crop, Ore, Whale } from '../tech/resource';

export default class Cell
{
  static idx: number = 0;
  static len: number = 0;

  ix: number;
  city: City;

  private _resource: Resource | null;
  private _struct: Structure | null;
  _type: string;
  _belongsTo: City | null = null;
  _unit: Unit | null = null;

  cacheAdj: { number: Cell[] } = {};

  constructor(type = 'field', resource: Resource | null = null)
  {
    this.ix = Cell.idx;
    this._type = type;
    this._resource = resource;
    Cell.idx++;
  }

  getResource(tribe: Tribe)
  {
    if (!tribe) return this._resource;
    if (!this._resource) return null;
    if (this._unit?.isFoe(tribe)) return null;

    if (this._resource.is(Fish))
    {
      if (!tribe.hasTech(Fishing.Fish, true))
      {
        return null;
      }
    }
    else if (this._resource.is(Crop))
    {
      if (!tribe.hasTech(Organization.Organize, true))
      {
        return null;
      }
    }
    else if (this._resource.is(Ore))
    {
      if (!tribe.hasTech(Climbing.Climb, true))
      {
        return null;
      }
    }
    else if (this._resource.is(Whale))
    {
      if (!tribe.hasTech(Fishing.Fish, true))
      {
        return null;
      }
    }

    return this._resource;
  }

  getStruct()
  {
    return this._struct;
  }

  setResource(resource: Resource)
  {
    this._resource = resource;
  }

  setStructure(struct)
  {
    this._struct = struct;
  }

  getTribe()
  {
    return this._belongsTo?._tribe;
  }

  getAdj(cells: Cell[], r: number | 'a' = 1, filter = (c) => true) {
    // TODO or is scout

    if (r === 'a') r = this.isMountain() ? 2 : 1;

    if (!this.cacheAdj[r])
    {
      this.cacheAdj[r] = Cell.getAdj(this.ix, r).map(i => cells[i]);
    }

    return this.cacheAdj[r].filter(filter);
  }

  getAdjTerritory(tribe, filter = (c) => true, r = 1)
  {
    let adj = this.cacheAdj[r];
    
    if (!adj) adj = Cell.getAdj(this.ix, r);
    
    return tribe.getTerritory().filter(c => filter(c) && adj.some(z => {
      c.is(z);
    }));
  }
  
  static cacheAdj: { number: { number: number[] } } = { };

  static getAdj(i, r = 1, includeCenter = false)
  {
    if(Cell.cacheAdj[i] && Cell.cacheAdj[i][r])
    {
      return Cell.cacheAdj[i][r];
    }
    
    const out = includeCenter ? [i] : [];
    const s = Math.sqrt(Cell.len);

    for (let x = -r; x <= r; x++)
    {
      if (Math.floor(i / s) !== Math.floor((x + i) / s))
      {
        continue;
      }

      for (let y = -r; y <= r; y++)
      {
        const _i = x + y * s + i;
        if (!x && !y) continue;
        if (_i < 0 || _i >= Cell.len) continue;
        out.push(_i);
      }
    }
    
    if(!Cell.cacheAdj[i]) Cell.cacheAdj[i] = {};
    
    Cell.cacheAdj[i][r] = out;
    
    return out;
  }
  
  // refs?
  setType(type): void {
    this._type = type;
  }

  is(other: Cell | number)
  {
    return this.ix === (other?.ix || other) + 0;
  }

  isTerritory()
  {
    return this._belongsTo || this.isCity();
  }

  isMine(tribe)
  {
    return tribe.is(this._belongsTo?._tribe);
  }

  isTown()
  {
    return this.isVillage() || this.isCity();
  }

  isForest()
  {
    return this._type === 'forest';
  }

  isMountain()
  {
    return this._type === 'mountain';
  }

  isWater()
  {
    return this._type === 'water';
  }

  isOcean()
  {
    return this._type === 'ocean';
  }

  isLiquid()
  {
    return this.isWater() || this.isOcean();
  }

  isVillage()
  {
    return this.city && !this.city._tribe;
  }

  isCity()
  {
    return this.city?._tribe;
  }

  isExplored(tribe)
  {
    return tribe.isExplored(this);
  }

  // ACTIONS

  // TODO prev changes untested

  // TODO incomplete, unused
  goResource()
  {

  }

  goStructure()
  {

  }
}