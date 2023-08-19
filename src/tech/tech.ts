export import Structure, { LumberHut, Sawmill, Mine, Forge, Farm, Windmill } from './structure';
export import Ability, { ClearForest } from './ability';
export import Task from './task';

import { Warrior, Rider, Swordsman, Archer, Defender, Knight, Catapult } from '/src/units';

export default class Tech
{
  id: string;
  tier: number;
  next: Tech[] | null;
  prev: Tech | null;
  unlocks: {
    ability: Ability
    struct: Structure,
    task: Task,
    unit: Unit, // ? class type NOT object
  };
  
  constructor (id: string, ...next: Tech[] | null = null)
  {
    this.id = id;
    this.next = next;
    this.tier = id.split('_')[0].length;
    this.next?.forEach(n => n.prev = this);
    this.unlocks = {};
  }
  
  getCache(): string {
    return [
      this.getTier()
    ].join(',');
  }
  
  // obsolete
  // ? retuns if can afford and is unlocked
  // bad practice
  isBuyable_(tribe)
  {
    return tribe._stars >= this.getCost(tribe) && this.isUnlocked(tribe);
  }
  
  // ? returns if this is equal to another
  is(other: Tech | string | null)
  {
    return this.id === (other?.id || other);
  }
  
  // ? returns if this is unlocked
  isUnlocked(tribe)
  {
    const [req, cost] = this.getRequired(tribe);
    return req.length < 2;
  }
  
  // ? returns cost to research this tech
  // ! if already owned: returns 0
  getCost(tribe, ignore=false)
  {
    // TODO sus..
    if(tribe.hasTech(this) && !ignore)
    {
      return 0;
    }
    const cost = tribe.getCities().length * this.tier + 4;
    // ! literacy 33% discount
    // TODO
    // vvv doesn't work for some reason
    return cost;// - tribe.hasTech(Climbing.Philosophy) ? Math.ceil(cost * 0.33) : 0;
  }
  
  // ? returns required tech to research and cost
  getRequired(tribe)
  {
    let cost = this.getCost(tribe);
    let required = cost? [this] : [];
    
    this.getPrevious().forEach(t => {
      const c = t.getCost(tribe);
      if(c) 
      {
        cost += c;
        required.push(t);
      }
    });
    
    return [required, cost];
  }
  
  getTier()
  {
    return this.tier;
  }
  
  // ? returns the lower branch
  getPrevious()
  {
    let prevs = [];
    let prev = this.prev;
    while (prev.prev)
    {
      prevs = [...prevs, prev];
      prev = prev.prev;
    }
    return prevs;
  }
  
  // ! unused
  getLast()
  {
    let last = this.prev;
    while(true)
    {
      if(!prev.prev) break;
      last = prev.prev;
    }
    return last;
  }
  
  setStruct(struct: Structure)
  {
    this.unlocks.struct = struct;
  }
  
  setAbility(ability: Ability)
  {
    this.unlocks.ability = ability;
  }
  
  setUnit(unit)
  {
    this.unlocks.unit = unit;
  }
  
  // TODO set task
  
  filter(cb: (t: Tech) => { })
  {
    const br = (next) => {
      if(!next) return [];
      
      let found = [];
      
      for (const t of next)
      {
        if (!cb(t)) continue;
        found = [...found, t, ...br(t.next)];
      }
      
      return found;
    };
    return br(this.next);
  }
}

const spiritualism = new Tech('spi_rit');
const archery = new Tech('ar_chery', spiritualism);
archery.setUnit(Archer);
const mathematics = new Tech('mat_hs');
mathematics.setStruct(Sawmill);
mathematics.setUnit(Catapult);
const forestry = new Tech('fo_restry', mathematics);
forestry.setAbility(ClearForest);
forestry.setStruct(LumberHut);
const hunting = new Tech('h_unt', forestry, archery);

const chivalry = new Tech('chi_v');
chivalry.setUnit(Knight);
const freeSpirit = new Tech('fr_ee', chivalry);
const trade = new Tech('tra_de');
const roads = new Tech('ro_ads', trade);
const riding = new Tech('r_ide', roads, freeSpirit);
riding.setUnit(Rider);

const navigation = new Tech('nav_');
const sailing = new Tech('sa_il', navigation);
const aquatism = new Tech('aqu_a');
const whaling = new Tech('wh_aling', aquatism);
const fishing = new Tech('f_ishing', whaling, sailing);

const diplomacy = new Tech('dip_lomacy');
const strategy = new Tech('st_rat', diplomacy);
strategy.setUnit(Defender);
const construction = new Tech('con_struct');
construction.setStruct(Windmill);
const farming = new Tech('fa_rm', construction);
farming.setStruct(Farm);
const organization = new Tech('o_rg', farming, strategy);

const philosophy = new Tech('phi_lo');
const meditation = new Tech('me_ditate', philosophy);
const smithery = new Tech('smi_thery');
smithery.setStruct(Forge);
smithery.setUnit(Swordsman);
const mining = new Tech('mi_ning', smithery);
mining.setStruct(Mine);
const climbing = new Tech('c_limb', mining, meditation);

export const Tree = new Tech('_', organization, climbing, fishing, hunting, riding);

export const enum Climbing {
  Climb = climbing,
  Mining = mining,
  Smithery = smithery,
  Meditation = meditation,
  Philosophy = philosophy
};

export const enum Organization {
  Organize = organization,
  Strategy = strategy,
  Diplomacy = diplomacy,
  Farming = farming,
  Construction = construction
};

export const enum Hunting {
  Hunt = hunting,
  Forestry = forestry,
  Mathematics = mathematics,
  Archery = archery,
  Spiritualism = spiritualism
};

export const enum Fishing {
  Fish = fishing,
  Sailing = sailing,
  Navigation = navigation,
  Whaling = whaling,
  Aquatism = aquatism
};

export const enum Riding {
  Ride = riding,
  Roads = roads,
  Trade = trade,
  FreeSpirit = freeSpirit,
  Chivalry = chivalry
};
