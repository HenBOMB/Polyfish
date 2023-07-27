import Tech, { Organization, Fishing, Climbing, Hunting } from './tech';
import Structure, { Farm, Mine } from './structure';

export default class Resource
{
  id: string;
  cost: number;
  required: Tech;
  struct: Structure | null;
  
  constructor(id: string, cost: number, pop: number, struct: Structure | null = null)
  {
    this.id = id;
    this.pop = pop;
    this.cost = cost;
    this.required = null;
    this.struct = struct;
  }
  
  getCost(_)
  {
    return this.cost;
  }
  
  setRequired(req)
  {
    this.required = req;
  }
  
  is(other: Resource | string | null)
  {
    return this.id === other?.id || this.id === other;
  }
} 

const Fruit = new Resource('fruit', 2, 1);

Fruit.setRequired(Organization.Organize);

const Crop = new Resource('crop', 5, 2, Farm);
Crop.setRequired(Organization.Farming);

const Fish = new Resource('fish', 2, 1);
Fish.setRequired(Fishing.Fish);

const Ore = new Resource('ore', 5, 2, Mine);
Ore.setRequired(Climbing.Mining);

const Game = new Resource('game', 2, 1);
Game.setRequired(Hunting.Hunt);

const Whale = new Resource('whale', 10, 0);
Whale.setRequired(Fishing.Whaling);

// const Ruins = new Resource('ruins', 0, 0);

export { Fruit, Crop, Fish, Ore, Game, Whale };