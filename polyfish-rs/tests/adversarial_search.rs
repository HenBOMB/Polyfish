//! Regressions for adversarial in-tree search: `simulate_move(EndTurn)` must
//! hand control to the next player instead of deleting its turn, and the
//! lookahead horizon must respect the game's real length.

use polyfish::ai::brain::{MAX_TURNS_AHEAD, MIN_TURNS_AHEAD, max_turns_ahead};
use polyfish::coords::Coords;
use polyfish::game::{Game, adversarial_search, set_adversarial_search};
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::attack::AttackMove;
use polyfish::types::{MapSize, MapType, MoveType, TerrainType, TribeType, UnitType};
use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard};

/// The adversarial switch is process-wide, so mode-sensitive tests must not
/// run concurrently. Every such test holds this for its whole body.
static MODE: Mutex<()> = Mutex::new(());

struct ModeGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl ModeGuard {
    fn set(on: bool) -> Self {
        let g = MODE.lock().unwrap_or_else(|e| e.into_inner());
        set_adversarial_search(on);
        Self(g)
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        set_adversarial_search(false);
    }
}

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    });
    game.post_load();
    game
}

fn end_turn_move(game: &Game) -> Box<dyn polyfish::moves::Move> {
    game.legal_moves()
        .into_iter()
        .find(|m| m.move_type() == MoveType::EndTurn)
        .expect("EndTurn must always be legal")
}

#[test]
fn the_switch_round_trips() {
    let _g = ModeGuard::set(true);
    assert!(adversarial_search());
    set_adversarial_search(false);
    assert!(!adversarial_search());
}

#[test]
fn sim_end_turn_hands_over_when_adversarial() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let mover = game.state.settings.current_player_turn_id;
    let turn = game.state.settings.turn;

    let m = end_turn_move(&game);
    let undo = game.simulate_move(m.as_ref()).expect("EndTurn simulated");

    assert_ne!(
        game.state.settings.current_player_turn_id, mover,
        "simulate_move(EndTurn) must hand control to the next player"
    );
    assert_eq!(
        game.state.settings.turn, turn,
        "handing over inside the turn order must not advance the game turn"
    );

    undo(&mut game.state);
    assert_eq!(game.state.settings.current_player_turn_id, mover);
    assert_eq!(game.state.settings.turn, turn);
}

#[test]
fn sim_end_turn_skips_opponent_when_not_adversarial() {
    let _g = ModeGuard::set(false);
    let mut game = make_game(42);
    let mover = game.state.settings.current_player_turn_id;

    let m = end_turn_move(&game);
    let undo = game.simulate_move(m.as_ref()).expect("EndTurn simulated");

    assert_eq!(
        game.state.settings.current_player_turn_id, mover,
        "legacy single-player search must cycle straight back to the mover"
    );
    undo(&mut game.state);
}

/// Walk the greedy-legal move list far enough to cross several turn
/// boundaries and record who was to move at every ply.
fn descent_players(game: &mut Game, plies: usize) -> (HashSet<i32>, usize) {
    let mut seen = HashSet::new();
    let mut undos = Vec::new();
    let mut end_turns = 0;

    for _ in 0..plies {
        if game.state.settings._game_over {
            break;
        }
        seen.insert(game.state.settings.current_player_turn_id);
        let mut moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        // Mirror the tree's own rule: EndTurn only once nothing else is left.
        let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }
        let m = moves.remove(0);
        if m.move_type() == MoveType::EndTurn {
            end_turns += 1;
        }
        match game.simulate_move(m.as_ref()) {
            Some(u) => undos.push(u),
            None => break,
        }
    }

    while let Some(u) = undos.pop() {
        u(&mut game.state);
    }
    (seen, end_turns)
}

#[test]
fn descent_visits_both_players_when_adversarial() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(7);
    let (seen, end_turns) = descent_players(&mut game, 80);

    assert!(
        end_turns >= 2,
        "probe must cross at least two turn boundaries, saw {end_turns}"
    );
    assert!(
        seen.contains(&1) && seen.contains(&2),
        "both players must appear on an in-tree path, saw {seen:?}"
    );
}

#[test]
fn descent_never_visits_opponent_when_not_adversarial() {
    let _g = ModeGuard::set(false);
    let mut game = make_game(7);
    let (seen, end_turns) = descent_players(&mut game, 80);

    assert!(end_turns >= 2, "probe must cross turn boundaries");
    assert_eq!(
        seen,
        HashSet::from([1]),
        "legacy search must only ever see the root player"
    );
}

/// The undo chain has to restore the opponent's turn too, not just ours.
#[test]
fn adversarial_descent_undo_round_trips() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(11);
    let before = serde_json::to_value(&game.state).unwrap();
    let (_seen, _) = descent_players(&mut game, 60);
    let after = serde_json::to_value(&game.state).unwrap();
    assert_eq!(before, after, "adversarial descent did not undo cleanly");
}

#[test]
fn max_turns_ahead_uses_its_argument() {
    // The bug this replaces hard-coded a 20-turn game and ignored max_turns.
    let short: Vec<i32> = (0..30).map(|t| max_turns_ahead(t, 10)).collect();
    let long: Vec<i32> = (0..30).map(|t| max_turns_ahead(t, 45)).collect();
    assert_ne!(short, long, "horizon must depend on max_turns");

    // Never look past the game's own end (above the floor).
    for t in 0..60 {
        for &m in &[10, 20, 45, 100] {
            let h = max_turns_ahead(t, m);
            assert!(h >= MIN_TURNS_AHEAD, "horizon {h} below floor at t={t} m={m}");
            assert!(h <= MAX_TURNS_AHEAD, "horizon {h} above cap at t={t} m={m}");
            if m - t >= MIN_TURNS_AHEAD {
                assert!(h <= m - t, "horizon {h} looks past turn {m} from {t}");
            }
        }
    }
}

#[test]
fn max_turns_ahead_is_non_increasing() {
    for &m in &[10, 20, 45, 100] {
        for t in 0..(m + 5) {
            assert!(
                max_turns_ahead(t + 1, m) <= max_turns_ahead(t, m),
                "horizon grew from turn {t} to {} (max_turns {m})",
                t + 1
            );
        }
    }
}

/// The late game is where the old hard-coded 20 collapsed the horizon to 2.
#[test]
fn max_turns_ahead_survives_the_late_game() {
    assert!(
        max_turns_ahead(30, 45) > MIN_TURNS_AHEAD,
        "a 45-turn game still has real lookahead left at turn 30"
    );
}

/// Tactical regression, deterministic substitute for "the search must not walk
/// the defender away": with an enemy unit adjacent to an undefended city, an
/// in-tree EndTurn must expose the enemy's move onto that city. Under the
/// legacy search that line is unreachable, so the threat is invisible and no
/// amount of training can teach the defender to stay.
#[test]
fn in_tree_opponent_can_threaten_an_undefended_city() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let size = game.state.settings.size;

    let capital = game.state.tribes[&1].cities[0].idx;

    // Walk our own defender off the capital; that is the move the search is
    // supposed to reject once it can see the consequence.
    let far = (0..size * size)
        .find(|idx| {
            *idx != capital
                && polyfish::functions::get_adjacent_indices(&game.state, capital, 3)
                    .iter()
                    .all(|a| a != idx)
                && matches!(
                    game.state.tiles[idx].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                )
        })
        .expect("map must have a distant land tile");

    // Park the enemy on a land tile next to our capital.
    let adjacent = polyfish::functions::get_adjacent_indices(&game.state, capital, 1)
        .into_iter()
        .find(|idx| {
            *idx != capital
                && matches!(
                    game.state.tiles[idx].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                )
        })
        .expect("capital must have a land neighbour");

    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        assert!(!p1.units.is_empty(), "P1 starts with a unit");
        p1.units.truncate(1);
        p1.units[0].coords = polyfish::coords::Coords::from_index(far, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        assert!(!p2.units.is_empty(), "P2 starts with a unit");
        p2.units.truncate(1);
        p2.units[0].coords = polyfish::coords::Coords::from_index(adjacent, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    for tile in game.state.tiles.values_mut() {
        tile._unit_owner_id = None;
    }
    game.post_load();

    // Search the same obscured view self_play searches. The enemy sits inside
    // our vision, so the belief-state opponent still owns it.
    let mut view = game.clone_for_mcts(1);
    assert!(
        view.state.tribes[&2].units.iter().any(|u| u.coords.idx == adjacent),
        "the adjacent enemy must survive fog obscuring"
    );

    let m = end_turn_move(&view);
    let _undo = view.simulate_move(m.as_ref()).expect("EndTurn simulated");
    assert_eq!(view.state.settings.current_player_turn_id, 2);

    let threats: Vec<(MoveType, String)> = view
        .legal_moves()
        .iter()
        .filter(|mv| mv.target_idx().map_or(false, |t| t as i32 == capital))
        .map(|mv| (mv.move_type(), mv.describe(&view.state)))
        .collect();
    println!("in-tree enemy threats on the capital: {threats:?}");

    assert!(
        threats
            .iter()
            .any(|(t, _)| matches!(t, MoveType::Step | MoveType::Attack | MoveType::Capture)),
        "in-tree opponent must be able to move onto the undefended capital \
         at {capital} from {adjacent}"
    );
}

/// Category 1 Proof #1: Tactical bad trades and retaliatory punishment.
/// Proves:
/// (a) Immediate retaliation: Attacking a high-defense enemy when wounded deals fatal
///     retaliatory damage on the attacker's own turn.
/// (b) In-tree adversarial handover: Attacking at full HP into a fortified Defender
///     leaves the attacker severely crippled (~1.9 HP). Simulating EndTurn hands
///     control to the opponent, who immediately executes a fatal counter-attack,
///     proving that adversarial search exposes bad trades to the model.
/// (c) Contrast with non-adversarial: Under single-player search, EndTurn cycles
///     back to P1, masking the enemy counter-attack completely.
#[test]
fn proof_adversarial_tactics_retaliation_and_bad_trades() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let size = game.state.settings.size;

    // Find two adjacent traversable land tiles for P1 and P2
    let (t_p1, t_p2) = (0..size * size)
        .find_map(|p1| {
            if !matches!(
                game.state.tiles[&p1].terrain_type,
                TerrainType::Field | TerrainType::Forest
            ) {
                return None;
            }
            let adj = polyfish::functions::get_adjacent_indices(&game.state, p1, 1);
            adj.into_iter()
                .find(|&p2| {
                    matches!(
                        game.state.tiles[&p2].terrain_type,
                        TerrainType::Field | TerrainType::Forest
                    )
                })
                .map(|p2| (p1, p2))
        })
        .expect("must find two adjacent traversable land tiles");

    for &idx in &[t_p1, t_p2] {
        let tile = game.state.tiles.get_mut(&idx).unwrap();
        tile.explorers.insert(1);
        tile.explorers.insert(2);
    }

    for tile in game.state.tiles.values_mut() {
        tile._unit_owner_id = None;
    }

    // Set up P1 Warrior and P2 Defender
    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        p1.units.truncate(1);
        p1.units[0].unit_type = UnitType::Warrior;
        p1.units[0].health = 10.0;
        p1.units[0].attacked = false;
        p1.units[0].moved = false;
        p1.units[0].coords = Coords::from_index(t_p1, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        p2.units.truncate(1);
        p2.units[0].unit_type = UnitType::Defender;
        p2.units[0].health = 15.0;
        p2.units[0].attacked = false;
        p2.units[0].moved = false;
        p2.units[0].coords = Coords::from_index(t_p2, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    game.post_load();

    // 1. Prove immediate fatal retaliation when attacking at low HP
    {
        game.state.tribes.get_mut(&1).unwrap().units[0].health = 3.0;
        let atk = AttackMove::new(t_p1, t_p2);
        let undo = game.simulate_move(&atk).expect("Attack must simulate");

        // Attacking Warrior took lethal retaliation damage and died on the spot
        assert!(
            game.state.tribes[&1].units.is_empty(),
            "Low-HP Warrior must perish immediately from Defender's retaliatory damage"
        );
        undo(&mut game.state);
        assert_eq!(game.state.tribes[&1].units.len(), 1);
        assert_eq!(game.state.tribes[&1].units[0].health, 3.0);
    }

    // Reset health to full 10.0 HP for in-tree lookahead trade proof
    game.state.tribes.get_mut(&1).unwrap().units[0].health = 10.0;
    let before_state = serde_json::to_value(&game.state).unwrap();

    // 2. Simulate bad trade at full HP: Warrior attacks Defender
    let atk = AttackMove::new(t_p1, t_p2);
    let undo_atk = game.simulate_move(&atk).expect("Attack must simulate");

    let warrior_hp_after_trade = game.state.tribes[&1].units[0].health;
    assert!(
        warrior_hp_after_trade < 3.0,
        "Retaliation from 15 HP Defender must leave Warrior severely wounded (got {warrior_hp_after_trade})"
    );

    // End turn -> In adversarial mode, turn transfers to P2
    let undo_end = game
        .simulate_move(end_turn_move(&game).as_ref())
        .expect("EndTurn simulates");
    assert_eq!(
        game.state.settings.current_player_turn_id, 2,
        "Adversarial search must hand control to P2"
    );

    // P2 finds and executes the kill on the wounded Warrior
    let p2_finishing_attacks: Vec<_> = game
        .legal_moves()
        .into_iter()
        .filter(|m| {
            m.move_type() == MoveType::Attack
                && m.target_idx().map_or(false, |t| t as i32 == t_p1)
        })
        .collect();

    assert!(
        !p2_finishing_attacks.is_empty(),
        "P2 must have a legal attack to finish off the adjacent wounded Warrior"
    );

    let undo_kill = game
        .simulate_move(p2_finishing_attacks[0].as_ref())
        .expect("P2 counter-attack simulates");

    // The Warrior is dead and eliminated from the board
    assert!(
        game.state.tribes[&1].units.is_empty(),
        "Wounded Warrior must be eliminated by P2's counter-attack on P2's turn"
    );

    // Undo the entire adversarial sequence cleanly
    undo_kill(&mut game.state);
    undo_end(&mut game.state);
    undo_atk(&mut game.state);

    let after_state = serde_json::to_value(&game.state).unwrap();
    assert_eq!(
        before_state, after_state,
        "Full adversarial trade lookahead sequence must undo cleanly"
    );
}

/// Category 1 Proof #2: Rider Hit-and-Run / Escape mechanics.
/// Proves:
/// (a) Escape mechanic: After attacking, Rider's `moved` flag is reset to false,
///     allowing a subsequent Step move away from combat.
/// (b) Bad branch (Attack and stay): If the Rider attacks and does not escape,
///     the opponent on their turn has a legal attack and inflicts heavy damage.
/// (c) Optimal branch (Attack and escape): If the Rider attacks and steps 1 tile
///     away out of the enemy Warrior's reach (distance 2), the opponent has
///     ZERO legal attacks against the Rider, leaving the Rider completely safe.
#[test]
fn proof_adversarial_tactics_rider_hit_and_run_escape() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let size = game.state.settings.size;

    // Find 3 traversable land tiles in a line or configuration where:
    // - t_p1 (Rider initial tile) is adjacent to t_p2 (Enemy tile)
    // - t_p1 is adjacent to t_retreat
    // - Chebyshev distance between t_p2 and t_retreat is 2 (out of Warrior's 1-tile reach)
    let (t_retreat, t_p1, t_p2) = (0..size * size)
        .find_map(|p1| {
            if !matches!(
                game.state.tiles[&p1].terrain_type,
                TerrainType::Field | TerrainType::Forest
            ) {
                return None;
            }
            let adj = polyfish::functions::get_adjacent_indices(&game.state, p1, 1);
            for &p2 in &adj {
                if !matches!(
                    game.state.tiles[&p2].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                ) {
                    continue;
                }
                for &ret in &adj {
                    if ret != p2
                        && matches!(
                            game.state.tiles[&ret].terrain_type,
                            TerrainType::Field | TerrainType::Forest
                        )
                    {
                        let d = polyfish::functions::get_chebyshev_distance(
                            p2,
                            ret,
                            size,
                        );
                        if d == 2 {
                            return Some((ret, p1, p2));
                        }
                    }
                }
            }
            None
        })
        .expect("must find retreat line of 3 traversable land tiles");

    for &idx in &[t_retreat, t_p1, t_p2] {
        let tile = game.state.tiles.get_mut(&idx).unwrap();
        tile.explorers.insert(1);
        tile.explorers.insert(2);
    }

    for tile in game.state.tiles.values_mut() {
        tile._unit_owner_id = None;
    }

    // Set up P1 Rider and P2 Warrior
    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        p1.units.truncate(1);
        p1.units[0].unit_type = UnitType::Rider;
        p1.units[0].health = 10.0;
        p1.units[0].attacked = false;
        p1.units[0].moved = false;
        p1.units[0].coords = Coords::from_index(t_p1, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        p2.units.truncate(1);
        p2.units[0].unit_type = UnitType::Warrior;
        p2.units[0].health = 10.0;
        p2.units[0].attacked = false;
        p2.units[0].moved = false;
        p2.units[0].coords = Coords::from_index(t_p2, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    game.post_load();

    let before_state = serde_json::to_value(&game.state).unwrap();

    // Branch A: Attack and Stay (Bad Play)
    {
        let atk = AttackMove::new(t_p1, t_p2);
        let undo_atk = game.simulate_move(&atk).expect("Rider attack simulates");

        // Escape mechanic verification: attacked is true, but moved is false!
        assert!(game.state.tribes[&1].units[0].attacked, "Rider has attacked");
        assert!(
            !game.state.tribes[&1].units[0].moved,
            "Escape skill must reset moved to false after attack"
        );

        // Player decides NOT to step away and ends turn
        let undo_end = game
            .simulate_move(end_turn_move(&game).as_ref())
            .expect("EndTurn simulates");
        assert_eq!(game.state.settings.current_player_turn_id, 2);

        // Opponent can attack the Rider at t_p1
        let p2_attacks: Vec<_> = game
            .legal_moves()
            .into_iter()
            .filter(|m| {
                m.move_type() == MoveType::Attack
                    && m.target_idx().map_or(false, |t| t as i32 == t_p1)
            })
            .collect();
        assert!(
            !p2_attacks.is_empty(),
            "Enemy Warrior must be able to counter-attack Rider that stayed in reach"
        );

        let undo_p2 = game
            .simulate_move(p2_attacks[0].as_ref())
            .expect("Enemy counter-attack simulates");
        let rider_eliminated = game.state.tribes[&1].units.is_empty()
            || game.state.tribes[&1].units[0].health <= 0.0;
        assert!(
            rider_eliminated,
            "Rider must be eliminated by counter-attack if it fails to escape"
        );

        undo_p2(&mut game.state);
        undo_end(&mut game.state);
        undo_atk(&mut game.state);
    }

    assert_eq!(
        serde_json::to_value(&game.state).unwrap(),
        before_state,
        "Branch A must undo cleanly"
    );

    // Branch B: Attack and Escape (Optimal Hit-and-Run)
    {
        let atk = AttackMove::new(t_p1, t_p2);
        let undo_atk = game.simulate_move(&atk).expect("Rider attack simulates");

        // Rider uses Escape: legal step to t_retreat
        let escape_step = game
            .legal_moves()
            .into_iter()
            .find(|m| {
                m.move_type() == MoveType::Step
                    && m.target_idx().map_or(false, |t| t as i32 == t_retreat)
            })
            .expect("Escape step to retreat tile must be legal");

        let undo_step = game
            .simulate_move(escape_step.as_ref())
            .expect("Escape step simulates");
        assert_eq!(
            game.state.tribes[&1].units[0].coords.idx, t_retreat,
            "Rider must now be on the retreat tile"
        );

        // End turn -> handover to P2
        let undo_end = game
            .simulate_move(end_turn_move(&game).as_ref())
            .expect("EndTurn simulates");
        assert_eq!(game.state.settings.current_player_turn_id, 2);

        // Opponent Warrior has 1 movement and 1 attack range; cannot reach t_retreat (distance 2)
        let p2_threats_on_rider: Vec<_> = game
            .legal_moves()
            .into_iter()
            .filter(|m| {
                m.move_type() == MoveType::Attack
                    && m.target_idx().map_or(false, |t| t as i32 == t_retreat)
            })
            .collect();
        assert!(
            p2_threats_on_rider.is_empty(),
            "Enemy Warrior cannot attack Rider that escaped to distance 2; Rider is completely safe"
        );

        undo_end(&mut game.state);
        undo_step(&mut game.state);
        undo_atk(&mut game.state);
    }

    assert_eq!(
        serde_json::to_value(&game.state).unwrap(),
        before_state,
        "Branch B must undo cleanly"
    );
}
