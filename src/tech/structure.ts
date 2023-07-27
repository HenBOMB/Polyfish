export default class Structure
{
  id: string;
  cost: number;
  pop: number;
  type: string;
  required: Resource;
  adjacent: Structure | null;

  constructor(id: string, cost: number, pop: number, type: string = 'field')
  {
    this.id = id;
    this.cost = cost;
    this.pop = pop;
    this.type = type;
  }
  
  getCost(_)
  {
    return this.cost;
  }

  is(other: Structure | string)
  {
    return this.id === (other?.id || other);
  }

  canBuild(cell, tribe)
  {
    if (cell._type !== this.type) return false;
    if (cell._struct) return false;
    if (this.required && !cell.getResource(tribe, true)?.is(this.required)) return false;
    if (this.adjacent) 
    {
      if (!cell.getAdjTerritory(tribe, (c) => this.adjacent.is(c._struct)).length)
      {
        return false;
      }
      
      // if already a struct like this in city
      if(cell._belongsTo.getTerritory().some(c => this.is(c._struct)))
      {
        return false;
      }
    }
    return true;
  }

  setAdjacent(struct)
  {
    this.adjacent = struct;
  }

  setRequired(required)
  {
    this.required = required;
  }
}

export const Farm = new Structure('farm', 5, 2);
Farm.setRequired('crop');

export const Windmill = new Structure('windmill', 5, 1);
Windmill.setAdjacent(Farm);

export const LumberHut = new Structure('lumber_hut', 2, 1, 'forest');
export const Sawmill = new Structure('sawmill', 5, 1);
Sawmill.setAdjacent(LumberHut);

export const Mine = new Structure('mine', 5, 2, 'mountain');
Mine.setRequired('ore');
export const Forge = new Structure('forge', 5, 2);
Forge.setAdjacent(Mine);