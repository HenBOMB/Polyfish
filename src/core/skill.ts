export default class Skill
{
  static Fortify = 'fortify';
  static Dash = 'dash';
  static Float = 'float';
  static Persist = 'persist';
  
  private id: string;

  constructor(id: string)
  {
    this.id = id;
  }
  
  is(other: Skill | string)
  {
    return this.id === (other?.id || other);
  }
}

export const Fortify = new Skill(Skill.Fortify);
export const Dash = new Skill(Skill.Dash);
export const Float = new Skill(Skill.Float);
export const Persist = new Skill(Skill.Persist);