export default class Ability
{
  id: string;
  type: string;
  stars: number;
  callback: () => {};
  required: (c) => {}

  constructor(id: string, stars: number = 0, type: string | null = null)
  {
    this.id = id;
    this.type = type;
    this.stars = stars;
    this.required = (c) => true;
  }

  canCast(c)
  {
    if (this.type && c._type !== this.type) return false;
    if (!this.required(c)) return false;
    return true;
  }

  cast(cell)
  {
    return this.callback(cell, cell.getTribe());
  }

  setCast(cb)
  {
    this.callback = cb;
  }

  setRequired(required)
  {
    this.required = required;
  }
}

// gives 1 star and transforms forest into field
export const ClearForest = new Ability('chop', 1, 'forest');
ClearForest.setRequired((cell) => {
  return !cell._struct;
});
ClearForest.setCast((cell, tribe) => {
  const bUndo = tribe.goBuy(-1);
  const old = cell.getResource();

  cell.setResource(null);
  cell.setField();
  
  return () => {
    bUndo();
    cell._type = 'forest';
    cell.setResource(old);
  };
});

export const Destroy = new Ability('destroy');
Destroy.setRequired((cell) => {
  return cell._struct;
});
Destroy.setCast((cell, tribe) => {
  const struct = cell.getStruct();

  // remove pop from cities
  // TODO a lot of work...

  // forge, sawmill..
  if (struct.adjacent)
  {

  }
  else
  {

  }

  cell.setStruct(null);

  return () => {
    cell.setStruct(struct);
  };
});