import Score from './score';
import Move, { Pass } from './move';
import { genEconomy } from './generator';

function deepClone(obj: any, clonedObjects = new WeakMap()): any {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (clonedObjects.has(obj)) {
    return clonedObjects.get(obj);
  }

  const clone = Array.isArray(obj) ? [] : Object.create(Object.getPrototypeOf(obj));

  clonedObjects.set(obj, clone);

  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const descriptor = Object.getOwnPropertyDescriptor(obj, key);
      if (descriptor && (typeof descriptor.value === 'function' || typeof descriptor.get === 'function')) {
        Object.defineProperty(clone, key, descriptor);
      } else {
        clone[key] = deepClone(obj[key], clonedObjects);
      }
    }
  }

  return clone;
}

class Action {
  move: Move;
  choices: string[];
  undo_: () => void | null;

  constructor(move: Move) {
    this.move = move;
    this.choices = [];
  }

  play(tribe): boolean {
    // TODO OBSOLETE
    const out = this.move.play(tribe);

    if (!out)
    {
      return false;
    }

    const [mUndo, rew = [], rews = []] = out.filter(Boolean);

    this.undo_ = mUndo;

    if (!mUndo)
    {
      return false;
    }

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
      if (r[1]) this.choices.push(r[1][0]);
      return claim(r);
    });

    this.undo_ = () => {
      mUndo();
      rUndo.forEach(r => r());
    }

    return true;
  }

  undo(): void {
    if (!this.undo_) return;
    this.undo_();
    this.undo_ = null;
  }
}

class State {
  tribe: any;
  moves: Move[];
  depth: number;
  action: Action | null;
  score: number;

  constructor(parent: State) {
    if (!parent) return;
    this.init(deepClone(parent.tribe), parent.depth-1);
  }

  init(tribe, depth): void {
    this.tribe = tribe;
    this.depth = depth;
    this.moves = genEconomy(tribe, true);
    this.score = Score(tribe);
  }

  clone(): State {
    this.revert();
    return deepClone(this);
  }
  
  revert(): void {
    this.action?.undo();
  }

  isTerminal(): boolean {
    return !this.depth || !this.moves.length;
  }

  // ! used in simulate() & expand()
  getNextState(action: Action): State {
    this.revert();
    
    if(action.play(this.tribe)) this.action = action;
    
    if(action.move.type !== 'skip') this.depth++;
   
    const state = new State(this);
    if(action.move.type === 'skip') this.depth--;
    
    this.revert();
    return state;
  }

  // ! used in simulate()
  getScore(): number {
    return this.score;
  }

  // ! used in isFullyExpanded()  
  getValidActions(): number {
    return this.moves.length;
  }

  // ! used in simulate() & expand()
  getRandomValidAction(): Action {
    let action = null;

    // ! skip is always last choice
    
    const _moves = [...this.moves];
    while (!action)
    {
      const i = Math.floor(Math.random() * _moves.length);
      action = new Action(_moves[i]);
      if (action.play(this.tribe)) break;
      else {
        _moves.splice(i, 1);
        action = null;
      }
    }
    
    action.undo();

    return action;
  }

  // ! used in getBestAction()
  getLastAction(): Action {
    return this.action;
  }
}

class Node {
  state: State;
  parent: Node | null;
  children: Node[];
  visits: number;
  score: number;

  constructor(state: State, parent: Node | null = null) {
    this.state = state;
    this.parent = parent;
    this.children = [];
    this.visits = 0;
    this.score = 0;
  }

  isFullyExpanded(): boolean {
    return this.children.length >= this.state.getValidActions();
  }

  selectChild(): Node {
    let selectedChild: Node;
    let bestUCT = -Infinity;

    for (const child of this.children) {
      const uct = child.score / child.visits + Math.sqrt(2 * Math.log(this.visits) / child.visits);
      if (uct > bestUCT) {
        selectedChild = child;
        bestUCT = uct;
      }
    }

    return selectedChild!;
  }

  expand(): Node {
    const action = this.state.getRandomValidAction();
    const nextState = this.state.getNextState(action);
    const child = new Node(nextState, this);
    this.children.push(child);
    return child;
  }

  simulate(): number {
    let state = this.state.clone();

    while (!state.isTerminal()) {
      const action = state.getRandomValidAction();
      state = state.getNextState(action);
    }

    return state.getScore();
  }

  backpropagate(result: number): void {
    this.visits++;
    this.score += result;
    this.state.revert();

    if (this.parent !== null) {
      this.parent.backpropagate(result);
    }
  }
}

class MCTS {
  root: Node;
  extent: number;

  constructor(tribe: any, depth: number, extent = 100) {
    const state = new State();
    state.init(deepClone(tribe), depth);
    this.root = new Node(state);
    this.extent = extent;
  }

  search(): void {
    for (let i = 0; i < this.extent; i++) {
      const selectedNode = this.selectNode();
      const expandedNode = this.expandNode(selectedNode);
      const result = this.simulate(expandedNode);
      this.backpropagate(expandedNode, result);
    }


    const gen = (state) => {
      return [state.score, state.tribe.getTtr(), state.action.move, state.tribe._stars, state.action.move.cell?.ix];
    };

    const nodes = [this.root, ...this.getPath()];
    const states = nodes.map(n => n.state);
    const moves = states.map(s => gen(s));

    return [this.root.score, 0, moves, states, this.extent];
  }

  selectNode(): Node {
    let currentNode = this.root;

    while (!currentNode.state.isTerminal()) {
      if (!currentNode.isFullyExpanded()) {
        return currentNode.expand();
      }

      currentNode = currentNode.selectChild();
    }

    return currentNode;
  }

  expandNode(node: Node): Node {
    if (node.state.isTerminal()) {
      return node;
    }
    return node.expand();
  }

  simulate(node: Node): number {
    return node.simulate();
  }

  backpropagate(node: Node, result: number): void {
    node.backpropagate(result);
  }

  getBestChild(children = this.root.children): State {
    let bestChild: Node | null = null;
    let bestScore = -Infinity;

    for (const child of children) {
      const score = child.score / child.visits;
      if (score > bestScore && child.state.getLastAction()) {
        bestChild = child;
        bestScore = score;
      }
    }

    return bestChild!;
  }

  getPath(children = this.root.children): Node[] {
    let bestChild = this.getBestChild(children);

    if (bestChild) {
      const branch: Node[] = [bestChild];
      const subBranch = this.getPath(bestChild.children);
      branch.push(...subBranch);
      return branch;
    }

    return [];
  }
}

export default MCTS;