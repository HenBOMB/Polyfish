import Unit from '/src/core/unit';
import { Fortify, Dash, Persist } from '/src/core/skill';

export default function getUnits()
{
  return {
    'warrior': Warrior,
    'rider': Rider,
    'archer': Archer,
    'defender': Defender,
    'swordsman': Swordsman,
    'knight': Knight,
    'catapult': Catapult
  }
}


// ? id, hp, atk, def, mov, range, cost

export const Warrior = (city) => {
  return new Unit('warrior', 10, 2, 2, 1, 1, 2, city, [Fortify, Dash]);
}

export const Rider = (city) => {
  return new Unit('rider', 10, 2, 1, 2, 1, 3, city, [Fortify, Dash]);
}

export const Archer = (city) => {
  return new Unit('archer', 10, 2, 1, 1, 2, 3, city, [Fortify, Dash]);
}

export const Defender = (city) => {
  return new Unit('defender', 15, 1, 3, 1, 1, 3, city, [Fortify, Dash]);
}

export const Swordsman = (city) => {
  return new Unit('swordsman', 15, 3, 3, 1, 1, 5, city, [Fortify, Dash]);
}

export const Knight = (city) => {
  return new Unit('knight', 10, 3.5, 1, 3, 1, 8, city, [Fortify, Dash, Persist]);
}

export const Catapult = (city) => {
  return new Unit('catapult', 10, 4, 0, 1, 3, 8, city, []);
}