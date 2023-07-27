import Tech from '../tech/tech';
import Cell from '../core/cell';
import Score from './score';
import { t2r } from './evaluate';

type Link = string | number;

export default class Move
{
  id: string;
  type: string;

  _undo: () => () => {} | null;
  _score: number;
  _ttr: number;
  _rewards: [][Link[]];

  constructor(id: string, type: string)
  {
    this.id = id;
    this.type = type;
    this._undo = null;
  }

  is(other: Move): boolean {
    return this.id === other.id;
  }

  go(tribe, out: any[], force = true): boolean {
    if (this._undo && !force)
    {
      this.undo();
    }

    const [mUndo, rew = [], rews = []] = Array.isArray(out) ? out : [out];

    if (!mUndo)
    {
      return false;
    }

    this._score = Score(tribe);
    this._ttr = t2r(tribe)
    this._undo = mUndo;
    this._rewards = [];

    const rewards = [...(rew.length ? [rew] : []), ...rews];

    if (!rewards.length)
    {
      return true;
    }

    const claim = (reward) => {
      const [undo, lab, choices] = reward;
      const undo_ = lab && choices[0]();

      return () => {
        undo_ && undo_();
        undo();
      };
    };

    const rUndo = rewards.map(r => {
      const o = claim(r);
      if (r[1]) this._rewards.push([Score(tribe), t2r(tribe), new Move(r[1][0], 'reward'), tribe._stars]);
      return o;
    });

    this._undo = () => {
      mUndo();
      rUndo.forEach(r => r());
    }

    return true;
  }

  undo(): void {
    if (!this._undo) return;
    this._undo();
    this._undo = null;
  }
}

export class Purchase extends Move
{
  tech: Tech;

  constructor(tech: Tech)
  {
    super(tech.id, 'research');
    this.tech = tech;
  }

  play(tribe): boolean {
    return this.go(tribe, tribe.goResearch(this.tech));
  }
}

export class Harvest extends Move
{
  cell: Cell;

  constructor(id: string, cell: any)
  {
    super(id, 'harvest');
    this.cell = cell;
  }

  play(tribe): boolean {
    return this.go(tribe, tribe.goHarvest(this.cell));
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.cell.ix}`,
    ];
  }
}

export class Capture extends Move
{
  city: any;
  cell: any;

  constructor(city: any)
  {
    super(city.isCity() ? 'city' : 'village', 'capture');
    this.city = city;
    this.cell = city.cell;
  }

  play(tribe): boolean {
    return this.go(tribe, this.city.goCapture());
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.cell.ix}`,
    ];
  }
}

export class Pass extends Move
{
  _turn: number;
  
  constructor()
  {
    super('pass', 'skip');
  }

  play(tribe): boolean {
    const out = this.go(tribe, tribe.goPass());
    this._turn = tribe._turn;
    return out;
  }
  
  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `turn: ${this._turn}`,
    ];
  }
}

export class Construct extends Move
{
  struct: Structure;
  cell: Cell;

  constructor(struct, cell)
  {
    super(struct.id, 'build');
    this.struct = struct;
    this.cell = cell;
  }

  play(tribe): boolean {
    return this.go(tribe, tribe.goBuild(this.cell, this.struct));
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.cell.ix}`,
    ];
  }
}

export class Cast extends Move
{
  abil: Ability;
  cell: Cell;

  constructor(abil, cell)
  {
    super(abil.id, 'ability');
    this.abil = abil;
    this.cell = cell;
  }

  play(tribe): boolean {
    return this.go(tribe, tribe.goCast(this.cell, this.abil));
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.cell.ix}`,
    ];
  }
}

export class Step extends Move
{
  unit: any;
  cell: any;

  _from: number;

  constructor(unit, cell)
  {
    super(unit.id, 'step');
    this.unit = unit;
    this.cell = cell;
  }

  play(tribe): boolean {
    this._from = this.unit._ix;
    return this.go(tribe, this.unit.goMove(this.cell));
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `from: ${this._from}`,
      `to: ${this.cell.ix}`,
    ];
  }
}

export class Attack extends Move
{
  unit: any;
  cell: any;

  constructor(unit, cell)
  {
    super(unit.id, 'attack');
    this.unit = unit;
    this.cell = cell;
  }

  play(tribe): boolean {
    return this.go(this.unit.goAttack(this.cell));
  }

  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.unit._ix}`,
      `foe: ${this.cell._unit._ix}`,
    ];
  }
}

export class Heal extends Move
{
  _health: number;

  constructor(unit)
  {
    super(unit.id, 'heal');
    this.unit = unit;
  }

  play(tribe): boolean {
    const g = this.go(tribe, this.unit.goHeal());
    this._health = this.unit._health;
    return g;
  }

  custom(): string[] {
    const hp = this._health - this.unit._health;
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.unit._ix}`,
      `hp: ${hp>0?'+':''}${hp}`
    ];
  }
}

export class Exaust extends Move
{
  unit: any;

  constructor(unit)
  {
    super(unit.id, 'wait');
    this.unit = unit;
  }

  play(tribe): boolean {
    return this.go(tribe, this.unit.goExaust(true));
  }
}

export class Hire extends Move
{
  unit: any;
  city: any;
  cell: any;

  constructor(unit, city)
  {
    super(unit.id, 'hire');
    this.unit = unit;
    this.city = city;
    this.cell = city.cell;
  }

  play(tribe): boolean {
    return this.go(tribe, this.city.goHire(this.unit));
  }
  
  custom(): string[] {
    return [
      `${this.id} (${this.type})`,
      `cell: ${this.unit._ix}`,
      `city: ${this.cell.ix}`,
    ];
  }
}