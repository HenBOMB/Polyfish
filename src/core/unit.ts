import Tribe from './tribe';
import City from './city';
import Skill from './skill';

export default class Unit
{
  id: string;
  attack: number;
  defense: number;
  range: number;
  movement: number;
  cost: number;
  tribe: Tribe;
  skills: Skill[];

  _ix: number = 0;
  _health: number;
  _city: City;
  _kills: number;
  _maxHealth: number;
  _veteran = false;
  _canMove = false;
  _canAttack = false;
  
  // TODO Push Rules
  // ! https://docs.google.com/document/u/0/d/1C3nQm6SnRFc5pkWMy_WOj8LRosuj2XlINCmCoDfswn4/mobilebasic
  _face: number;
  
  constructor(id: string, hp: number, atk: number, def: number, move: number, range: number, cost: number, city: City, skills: Skill[])
  {
    this.id = id;
    this.tribe = city?._tribe;
    this.attack = atk;
    this.defense = def;
    this._maxHealth = hp;
    this.range = range;
    this.movement = move;
    this.cost = cost;
    this._health = hp;
    this._kills = 0;
    this.skills = skills;
    this._city = city;
    
    if(city) city.goEnrole(this);
  }
  
  serialize(tribe: {}, city: {}): { } {
    
  }

  worth()
  {
    const cell = this.getCell();

    const dB = cell.isMountain() && this.tribe.hasTech('c_limb') ? 1.5 :
      cell.isLiquid() && this.tribe.hasTech('aqu_a') ? 1.5 :
      cell.isForest() && this.tribe.hasTech('ar_chery') ? 1.5 :
      cell.isCity() && this.hasSkill(Skill.fortify) ? cell.city.has('walls') ? 4 : 1.5 :
      1;

    const maxHp = this._maxHealth + (this._veteran ? 5 : 0);
    const dF = this.defense * (this._health / maxHp) * dB;
    const aF = this.attack * (this._health / maxHp);
    const m = this.movement;
    const r = this.range;

    return Math.ceil((dF + aF * r + m / 2) * 1000) / 1000;
  }

  is(other: Unit | number)
  {
    return this._ix === ((other?._ix || other) + 0);
  }

  isDead()
  {
    return this._health < 1;
  }

  isVeteran()
  {
    return this._veteran;
  }

  isPromotable()
  {
    return !this._veteran && this._kills > 2;
  }

  isSteppable(cell: Cell)
  {
    if (
      cell._unit ||
      cell.isLiquid() ||
      // port, sail needed
      (cell.getStruct()?.is('port') && !this.tribe.hasTech('sa_il')) ||
      // mountain, climb needed
      (cell.isMountain() && !this.tribe.hasTech('c_limb')) ||
      // ocean, nav needed
      (cell.isOcean() && !this.tribe.hasTech('nav_'))
    )
    {
      return false;
    }

    return cell.isExplored(this.tribe);
  }

  isReady(canAttack = false)
  {
    return (this._canMove && !canAttack) || (this._canAttack && canAttack);
  }
  
  isFoe(other: Tribe | null)
  {
    return !this.tribe.is(other);
  }

  getScore()
  {
    return this.cost * 5;
  }

  getCell()
  {
    return this.tribe.map.terrain[this._ix];
  }
  
  getCost(_)
  {
    return this.cost;
  }
  
  getVitality()
  {
    const m = this._maxHealth + (this._veteran ? 5 : 0);
    return this._health / m;
  }

  hasSkill(skill: Skill | string)
  {
    return this.skills.some(s => s.is(skill));
  }

  // ACTIONS
  
  // ? replenish action
  goReplenish()
  {
    const x = this._canMove, y = this._canAttack;
   
    const unit = this;

    unit._canMove = unit._canAttack = true;

    return () => {
      unit._canMove = x;
      unit._canAttack = y;
    }
  }

  goExaust(both: boolean)
  {
    const x = this._canMove, y = this._canAttack;
    
    if(!x && !y && both) return false;
    if(!x && !both) return false;
    
    const unit = this;

    unit._canMove = false;
    if(both) unit._canAttack = false;

    return () => {
      unit._canMove = x;
      if(both) unit._canAttack = y;
    }
  }

  goPromote()
  {
    if (this.isDead() || !this.isPromotable())
    {
      return false;
    }

    const unit = this;
    const hp = this._health;

    unit._maxHealth += 5;
    unit._health = unit._maxHealth;
    unit._veteran = true;

    return () => {
      unit._maxHealth -= 5;
      unit._health = hp;
      unit._veteran = false;
    }
  }
  
  goSpawn(cell: Cell)
  {
    const unit = this;
    const prev = this._ix;
    
    cell._unit = unit;
    unit._ix = cell.ix;
    
    return () => {
      cell._unit = null;
      unit._ix = prev;
    };
  }

  goDie()
  {
    if (this.isDead() || !this._city)
    {
      return false;
    }

    const remUndo = this._city.goExpel(this);
    const c = this.getCell();
    c._unit = null;

    return () => {
      remUndo();
    };
  }

  goHeal()
  {
    if (this.isDead() || !this._canMove || !this._canAttack || this._health === this._maxHealth)
    {
      return false;
    }

    const cell = this.getCell();
    const unit = this;
    const _health = unit._health;
    const health = unit._health + 2 + (unit.tribe.getTerritory().some(c => c.is(cell)) ? 2 : 0);

    const eUndo = this.goExaust(true);

    unit._health = health > unit._maxHealth ? unit._maxHealth : health;

    return () => {
      eUndo && eUndo();
      unit._health = _health;
    };
  }

  // ? starts an attack on another unit
  goAttack(cell)
  {
    const atk: Unit = this;
    const def: Unit = cell._unit;

    if (!this._canAttack)
    {
      return false;
    }

    if (atk.tribe.is(def.tribe))
    {
      return false;
    }

    const dB = cell.isMountain() && def.tribe.hasTech('c_limb') ? 1.5 :
      cell.isLiquid() && def.tribe.hasTech('aqu_a') ? 1.5 :
      cell.isForest() && def.tribe.hasTech('ar_chery') ? 1.5 :
      cell.isCity() && def.hasSkill(Skill.fortify) ? cell.city.has('walls') ? 4 : 1.5 :
      1;

    const aF = atk.attack * (atk._health / atk._maxHealth);

    const dF = def.defense * (def._health / def._maxHealth) * dB;

    const tD = aF + dF;

    const aR = Math.round((aF / tD) * atk.attack * 4.5);

    const aUndo = atk.goStrike(cell, aR);

    if (!aUndo)
    {
      return false;
    }

    if (def.isDead())
    {
      const mUndo = atk.goMove(cell);

      return () => {
        aUndo && aUndo();
        mUndo && mUndo();
      };
    }

    const exUndo = this.goExaust(true);

    const dR = Math.round((dF / tD) * def.defense * 4.5);

    const dUndo = def.goStrike(atk.getCell(), dR);

    return () => {
      aUndo && aUndo();
      dUndo && dUndo();
      exUndo && exUndo();
    };
  }
  
  // @exausts
  goMove(cell, stream = [null, 0]): () => {}
  {
    if (!this._canMove)
    {
      return false;
    }

    if (!this.isSteppable(cell))
    {
      return false;
    }

    const from = this.getCell();
    const unit = this;

    from._unit = null;
    unit._ix = cell.ix;
    cell._unit = unit;
    
    // ? hm..
    unit.tribe.foes.forEach(t => {
      if (t.isExplored(unit._ix)) return;
      t._memArmy.push(unit);
    });

    const eUndo = this.goExaust(!this.skills.some(s => s.is(Skill.Dash)));

    const [terr, radius] = stream;
    
    if (terr)
    {
      radius = radius ? radius : cell.isMountain() ? 2 : 1; // TODO scout?
      this.tribe.goExplore(cell, terr, radius);
      return false;
    }

    return () => {
      from._unit = unit;
      unit._ix = from.ix;
      cell._unit = null;
      eUndo && eUndo();
    };
  }

  // ? attacks an opponent
  goStrike(cell, force: number): () => {}
  {
    if (!cell._unit?.tribe.isExplored(cell))
    {
      return false;
    }

    const atk: Unit = this;
    const def: Unit = cell._unit;

    def._health -= force;

    if (def.isDead())
    {
      atk._kills++;
      const dUndo = def.goDie();

      return () => {
        atk._kills--;
        def._health += force;
        dUndo && dUndo();
      };
    }

    return () => {
      def._health += force;
    };
  }
}