import Tribe from '/src/core/tribe';
import City from '/src/core/city';
import { Organization, Climbing } from '/src/tech/tech';
import { Warrior, Swordsman } from './units';

function create(name, tech, unit, city, map)
{
  const tribe = new Tribe(name, tech);
  tribe.map = map;
  
  if (!city)
  {
    return tribe;
  }
  
  tribe._explored = [city.cell];
  city._tribe = tribe;
  const unit = unit(city);
  unit.goSpawn(city.cell);
  unit.goReplenish();
  city.cell._unit = null;
  unit.goMove(city.cell, [map.terrain, 2]);
  city._tribe = null;
  city.goCapture();
  tribe.addScore(unit.getScore());
  return tribe;
};

export const Imperius = (city, map) => {
  return create('imperius', Organization.Organize, Warrior, city, map);
};

export const Vengir = (city, map) => {
  return create('vengir', Climbing.Smithery, Swordsman, city, map);
};

export const XinXi = (city, map) => {
  return create('xin-xi', Climbing.Climb, Warrior, city, map);
};