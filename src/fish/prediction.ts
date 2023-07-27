import Tribe from '../core/tribe';
import Cell from '../core/cell';

// ? tries to predict where the villages are scattered, returns City[]
export function predictVillages(tribe: Tribe)
{
  const villages = [];
  
  // no more than 1 village will spawn in any 3x3 area
  // villages spawn one or two tiles away from resources (game, crop, fruit, ore)
  
  const territory = tribe.getTerritory();
  const likely = [];
  //const unlikely = [...territory];
  const uncharted = tribe.getExplored().filter(c=>!territory.some(cc=>c.is(cc)));
  
  uncharted.forEach(c => {
    if(c.isVillage())
    {
      villages.push(c);
    }
    else if(c.isForest() || c.getResource(tribe))
    {
      likely.push(c);
    }
    
    //unlikely.push(c);
  });
  
  const unknown = (c) => !c.isExplored(tribe);
  
  likely.forEach(c => {
    const adj = c.getAdj(tribe.map.terrain).filter(unknown);
    
    if(adj.length < 2)
    {
      return;
    }
    
    const m = Math.round(adj.reduce((p,c) => (p.ix||p)+(c.ix||c)) / adj.length);
    
    const mean = tribe.map.terrain[m];
    
    if(!villages.some(c=>c.is(mean)))
    {
      villages.push(mean);
    }
  });
  
  return villages;
}