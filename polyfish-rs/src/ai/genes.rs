//! AI Genes — Evolvable parameter set for the heuristic evaluator and move ordering.
//!
//! Instead of hardcoded constants, all tunable values live here as floats ("genes").
//! Evolution mutates and crosses these to discover optimal play strategies.
//!
//! Two presets exist: Perfection and Domination, since the game modes have
//! fundamentally different scoring and optimal strategies.

use crate::types::UnitType;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Top-Level Gene Container
// =============================================================================

/// Complete evolvable parameter set for the AI.
/// Every hardcoded constant in evaluator/*.rs and ordering.rs is captured here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIGenes {
    /// Move ordering weights
    pub ordering: OrderingGenes,
    /// Evaluator weights (player-level combination)
    pub evaluator: EvaluatorGenes,
    /// Economy evaluator weights
    pub economy: EconomyGenes,
    /// Army evaluator weights
    pub army: ArmyGenes,
    /// Expansion evaluator weights
    pub expansion: ExpansionGenes,
    /// Exploration evaluator weights
    pub exploration: ExplorationGenes,
    /// Game stage thresholds
    pub stages: StageGenes,
    /// MCTS parameters
    pub mcts: MctsGenes,
    /// Research (Technology) priorities
    pub research: ResearchGenes,
}

// =============================================================================
// Move Ordering Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingGenes {
    // Capture scores
    pub capture_ruin: f32,
    pub capture_village: f32,
    pub capture_city: f32,
    pub capture_starfish: f32,

    // Attack scores
    pub attack_kill: f32,
    pub attack_suicide: f32,
    pub attack_heavy_damage: f32,
    pub attack_light_damage: f32,
    pub attack_heavy_threshold: i32,

    // Ability scores
    pub ability_promote: f32,
    pub ability_combat: f32,      // Explode, Boost, FreezeArea, Convert
    pub ability_recover_critical: f32,
    pub ability_recover_safe: f32,
    pub ability_recover_waste: f32,
    pub ability_recover_critical_threshold: f32,
    pub ability_disband: f32,
    pub ability_burn_forest: f32,
    pub ability_destroy: f32,
    pub ability_default: f32,

    // Clear Forest scoring
    pub clear_forest_base: f32,
    pub clear_forest_resource_penalty: f32,
    pub clear_forest_forestry_penalty: f32,
    pub clear_forest_cluster_penalty_per: f32,
    pub clear_forest_enables_levelup_bonus: f32,
    pub clear_forest_desperation_bonus: f32,
    pub clear_forest_healthy_penalty: f32,

    // Summon scoring
    pub summon_early_penalty: f32,
    pub summon_base: f32,
    pub summon_threat_bonus: f32,
    pub summon_army_small_bonus: f32,
    pub summon_army_bloat_penalty: f32,
    pub summon_giant_bonus: f32,

    // Build / Harvest scoring
    pub build_base: f32,
    pub adjacency_lonely_penalty: f32,
    pub adjacency_2_bonus: f32,
    pub adjacency_3_bonus: f32,
    pub adjacency_4plus_bonus: f32,
    pub clustering_prereq_bonus: f32,
    pub levelup_completion_bonus: f32,
    pub levelup_miss_penalty: f32,

    // Temple timing (Perfection)
    pub temple_early_bonus: f32,
    pub temple_mid_bonus: f32,

    // Road scoring
    pub road_connection_unconnected_bonus: f32,
    pub road_connection_neither_bonus: f32,
    pub road_connection_both_bonus: f32,
    pub road_on_path_bonus: f32,
    pub road_adj_road_bonus: f32,
    pub road_adj_city_bonus: f32,
    pub road_single_city_penalty: f32,

    // Step scoring
    pub step_base: f32,
    pub step_capture_target_bonus: f32,
    pub step_enemy_city_bonus: f32,
    pub step_fog_reveal_bonus: f32,

    // Research scoring
    pub research_base: f32,
    pub research_buy_before_capture_bonus: f32,

    // Reward scoring
    pub reward_base: f32,
    pub reward_workshop_bonus: f32,
    pub reward_explorer_early_penalty: f32,
    pub reward_explorer_bonus: f32,
    pub reward_wall_threatened_bonus: f32,
    pub reward_wall_safe_bonus: f32,
    pub reward_resources_early_bonus: f32,
    pub reward_resources_late_bonus: f32,
    pub reward_pop_growth_bonus: f32,
    pub reward_border_growth_small_bonus: f32,
    pub reward_border_growth_large_bonus: f32,
    pub reward_park_perfection_bonus: f32,
    pub reward_park_domination_bonus: f32,
    pub reward_super_unit_perfection_bonus: f32,
    pub reward_super_unit_domination_bonus: f32,

    // Monument prioritization
    #[serde(default)]
    pub monument_bonus: f32,
}

// =============================================================================
// Evaluator Genes (player.rs weights)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorGenes {
    // Early game weights (Perfection)
    #[serde(alias = "early_eco", default)]
    pub early_perf_eco: f32,
    #[serde(alias = "early_mil", default)]
    pub early_perf_mil: f32,
    #[serde(alias = "early_exp", default)]
    pub early_perf_exp: f32,
    #[serde(alias = "early_fow", default)]
    pub early_perf_fow: f32,

    // Early game weights (Domination)
    #[serde(alias = "early_eco", default)]
    pub early_dom_eco: f32,
    #[serde(alias = "early_mil", default)]
    pub early_dom_mil: f32,
    #[serde(alias = "early_exp", default)]
    pub early_dom_exp: f32,
    #[serde(alias = "early_fow", default)]
    pub early_dom_fow: f32,

    // Mid game weights (Perfection)
    #[serde(default)]
    pub mid_perf_eco: f32,
    #[serde(default)]
    pub mid_perf_mil: f32,
    #[serde(default)]
    pub mid_perf_exp: f32,
    #[serde(default)]
    pub mid_perf_fow: f32,

    // Mid game weights (Domination)
    #[serde(default)]
    pub mid_dom_eco: f32,
    #[serde(default)]
    pub mid_dom_mil: f32,
    #[serde(default)]
    pub mid_dom_exp: f32,
    #[serde(default)]
    pub mid_dom_fow: f32,

    // End game weights (Perfection)
    #[serde(default)]
    pub end_perf_eco: f32,
    #[serde(default)]
    pub end_perf_mil: f32,
    #[serde(default)]
    pub end_perf_exp: f32,
    #[serde(default)]
    pub end_perf_fow: f32,

    // End game weights (Domination)
    #[serde(default)]
    pub end_dom_eco: f32,
    #[serde(default)]
    pub end_dom_mil: f32,
    #[serde(default)]
    pub end_dom_exp: f32,
    #[serde(default)]
    pub end_dom_fow: f32,
}

// =============================================================================
// Economy Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomyGenes {
    pub income_weight: f32,
    pub stars_weight: f32,
    pub tech_weight: f32,
    #[serde(default)]
    pub score_weight: f32,

    pub partial_city_perf_bonus_per: f32,
    pub partial_city_perf_cap: f32,
    pub partial_city_dom_penalty_per: f32,
    pub partial_city_dom_cap: f32,

    pub unused_tech_resource_penalty: f32,
    pub unused_tech_struct_no_terrain_penalty: f32,
    pub unused_tech_struct_have_terrain_penalty: f32,
    pub unused_tech_chain_penalty: f32,
    pub unused_tech_adjacency_penalty: f32,
    pub unused_tech_cap: f32,

    pub bad_struct_lonely_penalty: f32,
    pub bad_struct_cluster_reward: f32,
    pub bad_struct_crowded_penalty: f32,
    pub bad_struct_dead_road_penalty: f32,
    pub bad_struct_cap_min: f32,
    pub bad_struct_cap_max: f32,

    pub low_stars_threshold: f32,
    pub low_stars_max_penalty: f32,

    pub capital_connection_max_bonus: f32,

    pub urgency_weight: f32,
}

// =============================================================================
// Army Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmyGenes {
    /// Base weight for unit type value
    pub base_weight: f32,
    /// Weight for HP fraction
    pub hp_weight: f32,
    /// Weight for status effects
    pub status_weight: f32,
    /// Weight for defense bonus
    pub defense_weight: f32,

    /// Loneliness penalty (no friends nearby)
    pub loneliness_no_friends: f32,
    /// Loneliness penalty (only 1 friend)
    pub loneliness_one_friend: f32,
    /// Threshold base score for loneliness check
    pub loneliness_threshold: f32,

    /// Status bonuses/penalties
    pub veteran_bonus: f32,
    pub boost_bonus: f32,
    pub kill_bonus_per: f32,
    pub poison_penalty: f32,
    pub frozen_penalty: f32,

    /// Per-unit type value scores (the 46 unit values)
    pub unit_values: HashMap<String, f32>,

    /// Army size scaling per game phase
    pub early_units_per_city: f32,
    pub early_extra_units: f32,
    pub mid_units_per_city: f32,
    pub mid_extra_units: f32,
    pub late_units_per_city: f32,
    pub late_extra_units: f32,
}

// =============================================================================
// Expansion & Exploration Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionGenes {
    pub village_capture_bonus: f32,
    pub enemy_city_capture_bonus: f32,
    pub city_count_normalizer: f32,
    pub level_normalizer: f32,
    pub city_count_weight: f32,
    pub level_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationGenes {
    pub max_exploration_target: f32,
}

// =============================================================================
// Stage Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageGenes {
    /// Progress fraction threshold for early→mid transition
    pub early_threshold: f32,
    /// Progress fraction threshold for mid→late transition
    pub late_threshold: f32,
}

// =============================================================================
// MCTS Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsGenes {
    pub exploration_constant: f32,
    pub max_rollout_depth: usize,
}

// =============================================================================
// Research Genes
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchGenes {
    // Resource multipliers
    pub org_fruit_multiplier: f32,
    pub hunting_game_multiplier: f32,
    pub fishing_fish_multiplier: f32,
    pub farming_crop_multiplier: f32,
    pub mining_metal_multiplier: f32,

    // Terrain multipliers
    pub forestry_forest_multiplier: f32,
    pub climbing_mountain_multiplier: f32,
    pub sailing_water_multiplier: f32,
    pub navigation_ocean_multiplier: f32,

    // Military tech base values
    pub riding_base: f32,
    pub riding_field_multiplier: f32,
    pub archery_base: f32,
    pub strategy_base: f32,
    pub chivalry_base: f32,
    pub smithery_base: f32,

    // Infrastructure tech base values
    pub roads_per_city_multiplier: f32,
    pub trade_customs_multiplier: f32,

    // Others
    pub philosophy_per_tech_multiplier: f32,
    pub diplomacy_per_player_multiplier: f32,

    // Cost offsets
    pub tier_1_cost_offset: f32,
    pub tier_2_cost_offset: f32,
    pub tier_3_cost_offset: f32,
}

// =============================================================================
// Default Implementation (Generation 0 = current hardcoded values)
// =============================================================================

impl Default for AIGenes {
    fn default() -> Self {
        Self {
            ordering: OrderingGenes::default(),
            evaluator: EvaluatorGenes::default(),
            economy: EconomyGenes::default(),
            army: ArmyGenes::default(),
            expansion: ExpansionGenes::default(),
            exploration: ExplorationGenes::default(),
            stages: StageGenes::default(),
            mcts: MctsGenes::default(),
            research: ResearchGenes::default(),
        }
    }
}

impl Default for ResearchGenes {
    fn default() -> Self {
        Self {
            org_fruit_multiplier: 2.5,
            hunting_game_multiplier: 2.5,
            fishing_fish_multiplier: 2.5,
            farming_crop_multiplier: 3.0,
            mining_metal_multiplier: 4.0,

            forestry_forest_multiplier: 1.5,
            climbing_mountain_multiplier: 1.0,
            sailing_water_multiplier: 2.0,
            navigation_ocean_multiplier: 2.5,

            riding_base: 2.0,
            riding_field_multiplier: 0.2,
            archery_base: 1.5,
            strategy_base: 2.0,
            chivalry_base: 5.0,
            smithery_base: 6.0,

            roads_per_city_multiplier: 0.5,
            trade_customs_multiplier: 1.0,

            philosophy_per_tech_multiplier: 0.3,
            diplomacy_per_player_multiplier: 1.5,

            tier_1_cost_offset: 2.0,
            tier_2_cost_offset: 3.0,
            tier_3_cost_offset: 5.0, // Chivalry is 6 in raw code, Navigation is 5.
        }
    }
}

impl Default for OrderingGenes {
    fn default() -> Self {
        Self {
            capture_ruin: 100.0,
            capture_village: 99.8,
            capture_city: 100.1,
            capture_starfish: 80.0,

            attack_kill: 45.0,
            attack_suicide: 1.0,
            attack_heavy_damage: 25.0,
            attack_light_damage: 15.0,
            attack_heavy_threshold: 5,

            ability_promote: 35.0,
            ability_combat: 20.0,
            ability_recover_critical: 40.0,
            ability_recover_safe: 30.0,
            ability_recover_waste: 5.0,
            ability_recover_critical_threshold: 0.4,
            ability_disband: -50.0,
            ability_burn_forest: 5.0,
            ability_destroy: -10.0,
            ability_default: 10.0,

            clear_forest_base: 3.0,
            clear_forest_resource_penalty: -50.0,
            clear_forest_forestry_penalty: -10.0,
            clear_forest_cluster_penalty_per: 2.5,
            clear_forest_enables_levelup_bonus: 10.0,
            clear_forest_desperation_bonus: 2.0,
            clear_forest_healthy_penalty: -10.0,

            summon_early_penalty: -10.0,
            summon_base: 10.0,
            summon_threat_bonus: 15.0,
            summon_army_small_bonus: 8.0,
            summon_army_bloat_penalty: -15.0,
            summon_giant_bonus: 15.0,

            build_base: 22.0,
            adjacency_lonely_penalty: -2.0,
            adjacency_2_bonus: 5.0,
            adjacency_3_bonus: 12.0,
            adjacency_4plus_bonus: 18.0,
            clustering_prereq_bonus: 2.5,
            levelup_completion_bonus: 5.0,
            levelup_miss_penalty: -4.0,

            temple_early_bonus: 15.0,
            temple_mid_bonus: 8.0,

            road_connection_unconnected_bonus: 8.0,
            road_connection_neither_bonus: 4.0,
            road_connection_both_bonus: 1.0,
            road_on_path_bonus: 5.0,
            road_adj_road_bonus: 2.0,
            road_adj_city_bonus: 3.0,
            road_single_city_penalty: -3.0,

            step_base: 50.0,
            step_capture_target_bonus: 40.0,
            step_enemy_city_bonus: 45.0,
            step_fog_reveal_bonus: 2.0,

            research_base: 8.0,
            research_buy_before_capture_bonus: 5.0,

            reward_base: 200.0,
            reward_workshop_bonus: 10.0,
            reward_explorer_early_penalty: 3.0,
            reward_explorer_bonus: 5.0,
            reward_wall_threatened_bonus: 12.0,
            reward_wall_safe_bonus: 4.0,
            reward_resources_early_bonus: 9.0,
            reward_resources_late_bonus: 6.0,
            reward_pop_growth_bonus: 8.0,
            reward_border_growth_small_bonus: 9.0,
            reward_border_growth_large_bonus: 5.0,
            reward_park_perfection_bonus: 20.0,
            reward_park_domination_bonus: 5.0,
            reward_super_unit_perfection_bonus: 8.0,
            reward_super_unit_domination_bonus: 18.0,
            monument_bonus: 40.0,
        }
    }
}

impl Default for EvaluatorGenes {
    fn default() -> Self {
        Self {
            early_perf_eco: 0.5,
            early_perf_mil: 0.05,
            early_perf_exp: 0.3,
            early_perf_fow: 0.15,

            early_dom_eco: 0.4,
            early_dom_mil: 0.1,
            early_dom_exp: 0.35,
            early_dom_fow: 0.15,

            mid_perf_eco: 0.4,
            mid_perf_mil: 0.1,
            mid_perf_exp: 0.3,
            mid_perf_fow: 0.2,

            mid_dom_eco: 0.1,
            mid_dom_mil: 0.4,
            mid_dom_exp: 0.1,
            mid_dom_fow: 0.4,

            end_perf_eco: 0.4,
            end_perf_mil: 0.1,
            end_perf_exp: 0.25,
            end_perf_fow: 0.25,

            end_dom_eco: 0.2,
            end_dom_mil: 0.4,
            end_dom_exp: 0.2,
            end_dom_fow: 0.2,
        }
    }
}

impl Default for EconomyGenes {
    fn default() -> Self {
        Self {
            income_weight: 0.5,
            stars_weight: 0.1,
            tech_weight: 0.2,
            score_weight: 0.2,

            partial_city_perf_bonus_per: 0.03,
            partial_city_perf_cap: 0.10,
            partial_city_dom_penalty_per: 0.05,
            partial_city_dom_cap: 0.15,

            unused_tech_resource_penalty: 0.01,
            unused_tech_struct_no_terrain_penalty: 0.03,
            unused_tech_struct_have_terrain_penalty: 0.02,
            unused_tech_chain_penalty: 0.025,
            unused_tech_adjacency_penalty: 0.015,
            unused_tech_cap: 0.15,

            bad_struct_lonely_penalty: 0.02,
            bad_struct_cluster_reward: 0.02,
            bad_struct_crowded_penalty: 0.02,
            bad_struct_dead_road_penalty: 0.01,
            bad_struct_cap_min: -0.06,
            bad_struct_cap_max: 0.12,

            low_stars_threshold: 8.0,
            low_stars_max_penalty: 0.25,

            capital_connection_max_bonus: 0.10,

            urgency_weight: 0.1,
        }
    }
}

impl Default for ArmyGenes {
    fn default() -> Self {
        let mut unit_values = HashMap::new();

        // S-tier (0.70-0.95)
        unit_values.insert("FireDragon".into(), 0.95);
        unit_values.insert("BabyDragon".into(), 0.80);
        unit_values.insert("Crab".into(), 0.85);
        unit_values.insert("Giant".into(), 0.80);
        unit_values.insert("Juggernaut".into(), 0.78);
        unit_values.insert("Centipede".into(), 0.75);
        unit_values.insert("LivingIsland".into(), 0.70);
        unit_values.insert("Shaman".into(), 0.70);

        // A-tier (0.50-0.69)
        unit_values.insert("Knight".into(), 0.65);
        unit_values.insert("Tridention".into(), 0.65);
        unit_values.insert("Rider".into(), 0.60);
        unit_values.insert("Doomux".into(), 0.58);
        unit_values.insert("Amphibian".into(), 0.55);
        unit_values.insert("Hexapod".into(), 0.55);
        unit_values.insert("Rammer".into(), 0.55);
        unit_values.insert("Bomber".into(), 0.55);
        unit_values.insert("Mantis".into(), 0.55);
        unit_values.insert("Catapult".into(), 0.52);
        unit_values.insert("Archer".into(), 0.50);
        unit_values.insert("Scout".into(), 0.50);
        unit_values.insert("Exida".into(), 0.50);

        // B-tier (0.30-0.49)
        unit_values.insert("Swordsman".into(), 0.48);
        unit_values.insert("Cloak".into(), 0.47);
        unit_values.insert("Boomchi".into(), 0.45);
        unit_values.insert("Moth".into(), 0.45);
        unit_values.insert("Defender".into(), 0.43);
        unit_values.insert("Pirate".into(), 0.42);
        unit_values.insert("Dagger".into(), 0.40);
        unit_values.insert("Polytaur".into(), 0.40);
        unit_values.insert("Phychi".into(), 0.40);
        unit_values.insert("Raychi".into(), 0.40);
        unit_values.insert("Warrior".into(), 0.38);
        unit_values.insert("Segment".into(), 0.35);
        unit_values.insert("Kiton".into(), 0.35);
        unit_values.insert("Dinghy".into(), 0.33);
        unit_values.insert("MindBender".into(), 0.30);

        // C-tier (0.05-0.29)
        unit_values.insert("DragonEgg".into(), 0.25);
        unit_values.insert("Larva".into(), 0.20);
        unit_values.insert("Raft".into(), 0.15);
        unit_values.insert("InsectEgg".into(), 0.10);

        // Disabled (Polaris)
        unit_values.insert("Gaami".into(), 0.05);
        unit_values.insert("Mooni".into(), 0.05);
        unit_values.insert("BattleSled".into(), 0.05);
        unit_values.insert("IceArcher".into(), 0.05);
        unit_values.insert("IceFortress".into(), 0.05);

        Self {
            base_weight: 0.4,
            hp_weight: 0.3,
            status_weight: 0.2,
            defense_weight: 0.1,

            loneliness_no_friends: 0.15,
            loneliness_one_friend: 0.05,
            loneliness_threshold: 0.6,

            veteran_bonus: 0.2,
            boost_bonus: 0.15,
            kill_bonus_per: 0.05,
            poison_penalty: 0.2,
            frozen_penalty: 0.4,

            unit_values,

            early_units_per_city: 2.0,
            early_extra_units: 1.0,
            mid_units_per_city: 2.0,
            mid_extra_units: 2.0,
            late_units_per_city: 4.0,
            late_extra_units: 4.0,
        }
    }
}

impl Default for ExpansionGenes {
    fn default() -> Self {
        Self {
            village_capture_bonus: 0.5,
            enemy_city_capture_bonus: 0.4,
            city_count_normalizer: 9.0,
            level_normalizer: 30.0,
            city_count_weight: 0.5,
            level_weight: 0.5,
        }
    }
}

impl Default for ExplorationGenes {
    fn default() -> Self {
        Self {
            max_exploration_target: 0.8,
        }
    }
}

impl Default for StageGenes {
    fn default() -> Self {
        Self {
            early_threshold: 0.3,
            late_threshold: 0.7,
        }
    }
}

impl Default for MctsGenes {
    fn default() -> Self {
        Self {
            exploration_constant: 0.6,
            max_rollout_depth: 30,
        }
    }
}

// =============================================================================
// Unit Value Lookup (connects String keys to UnitType)
// =============================================================================

impl ArmyGenes {
    /// Get the gene value for a specific unit type.
    /// Falls back to 0.1 if the unit type is not in the gene map.
    pub fn get_unit_value(&self, unit_type: UnitType) -> f32 {
        let key = format!("{:?}", unit_type);
        *self.unit_values.get(&key).unwrap_or(&0.1)
    }
}

// =============================================================================
// Evolution: Mutation & Crossover
// =============================================================================

impl AIGenes {
    /// Save genes to a JSON file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load genes from a JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Create a mutated copy of these genes.
    /// Each float gene is perturbed by a random amount within ±rate (as a fraction).
    pub fn mutate(&self, rate: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut child = self.clone();

        // Helper closure to mutate a single f32 gene
        let mutf = |val: f32, rng: &mut rand::rngs::ThreadRng| -> f32 {
            if rng.r#gen::<f32>() < 0.3 {
                // 30% chance to mutate each gene
                let delta = val * rate * rng.gen_range(-1.0..1.0);
                val + delta
            } else {
                val
            }
        };

        // Mutate ordering genes
        child.ordering.capture_ruin = mutf(child.ordering.capture_ruin, &mut rng);
        child.ordering.capture_village = mutf(child.ordering.capture_village, &mut rng);
        child.ordering.capture_city = mutf(child.ordering.capture_city, &mut rng);
        child.ordering.attack_kill = mutf(child.ordering.attack_kill, &mut rng);
        child.ordering.attack_suicide = mutf(child.ordering.attack_suicide, &mut rng);
        child.ordering.attack_heavy_damage = mutf(child.ordering.attack_heavy_damage, &mut rng);
        child.ordering.attack_light_damage = mutf(child.ordering.attack_light_damage, &mut rng);
        child.ordering.summon_base = mutf(child.ordering.summon_base, &mut rng);
        child.ordering.summon_threat_bonus = mutf(child.ordering.summon_threat_bonus, &mut rng);
        child.ordering.summon_giant_bonus = mutf(child.ordering.summon_giant_bonus, &mut rng);
        child.ordering.build_base = mutf(child.ordering.build_base, &mut rng);
        child.ordering.adjacency_2_bonus = mutf(child.ordering.adjacency_2_bonus, &mut rng);
        child.ordering.adjacency_3_bonus = mutf(child.ordering.adjacency_3_bonus, &mut rng);
        child.ordering.adjacency_4plus_bonus = mutf(child.ordering.adjacency_4plus_bonus, &mut rng);
        child.ordering.clustering_prereq_bonus = mutf(child.ordering.clustering_prereq_bonus, &mut rng);
        child.ordering.levelup_completion_bonus = mutf(child.ordering.levelup_completion_bonus, &mut rng);
        child.ordering.step_base = mutf(child.ordering.step_base, &mut rng);
        child.ordering.step_fog_reveal_bonus = mutf(child.ordering.step_fog_reveal_bonus, &mut rng);
        child.ordering.research_base = mutf(child.ordering.research_base, &mut rng);
        child.ordering.research_buy_before_capture_bonus = mutf(child.ordering.research_buy_before_capture_bonus, &mut rng);
        child.ordering.reward_workshop_bonus = mutf(child.ordering.reward_workshop_bonus, &mut rng);
        child.ordering.reward_park_perfection_bonus = mutf(child.ordering.reward_park_perfection_bonus, &mut rng);
        child.ordering.reward_park_domination_bonus = mutf(child.ordering.reward_park_domination_bonus, &mut rng);
        child.ordering.reward_super_unit_perfection_bonus = mutf(child.ordering.reward_super_unit_perfection_bonus, &mut rng);
        child.ordering.reward_super_unit_domination_bonus = mutf(child.ordering.reward_super_unit_domination_bonus, &mut rng);
        child.ordering.monument_bonus = mutf(child.ordering.monument_bonus, &mut rng);

        // Mutate evaluator weights
        child.evaluator.early_perf_eco = mutf(child.evaluator.early_perf_eco, &mut rng);
        child.evaluator.early_perf_mil = mutf(child.evaluator.early_perf_mil, &mut rng);
        child.evaluator.early_perf_exp = mutf(child.evaluator.early_perf_exp, &mut rng);
        child.evaluator.early_perf_fow = mutf(child.evaluator.early_perf_fow, &mut rng);
        child.evaluator.early_dom_eco = mutf(child.evaluator.early_dom_eco, &mut rng);
        child.evaluator.early_dom_mil = mutf(child.evaluator.early_dom_mil, &mut rng);
        child.evaluator.early_dom_exp = mutf(child.evaluator.early_dom_exp, &mut rng);
        child.evaluator.early_dom_fow = mutf(child.evaluator.early_dom_fow, &mut rng);

        child.evaluator.mid_perf_eco = mutf(child.evaluator.mid_perf_eco, &mut rng);
        child.evaluator.mid_perf_mil = mutf(child.evaluator.mid_perf_mil, &mut rng);
        child.evaluator.mid_perf_exp = mutf(child.evaluator.mid_perf_exp, &mut rng);
        child.evaluator.mid_perf_fow = mutf(child.evaluator.mid_perf_fow, &mut rng);
        child.evaluator.mid_dom_eco = mutf(child.evaluator.mid_dom_eco, &mut rng);
        child.evaluator.mid_dom_mil = mutf(child.evaluator.mid_dom_mil, &mut rng);
        child.evaluator.mid_dom_exp = mutf(child.evaluator.mid_dom_exp, &mut rng);
        child.evaluator.mid_dom_fow = mutf(child.evaluator.mid_dom_fow, &mut rng);

        child.evaluator.end_perf_eco = mutf(child.evaluator.end_perf_eco, &mut rng);
        child.evaluator.end_perf_mil = mutf(child.evaluator.end_perf_mil, &mut rng);
        child.evaluator.end_perf_exp = mutf(child.evaluator.end_perf_exp, &mut rng);
        child.evaluator.end_perf_fow = mutf(child.evaluator.end_perf_fow, &mut rng);
        child.evaluator.end_dom_eco = mutf(child.evaluator.end_dom_eco, &mut rng);
        child.evaluator.end_dom_mil = mutf(child.evaluator.end_dom_mil, &mut rng);
        child.evaluator.end_dom_exp = mutf(child.evaluator.end_dom_exp, &mut rng);
        child.evaluator.end_dom_fow = mutf(child.evaluator.end_dom_fow, &mut rng);

        // Mutate economy genes
        child.economy.income_weight = mutf(child.economy.income_weight, &mut rng);
        child.economy.stars_weight = mutf(child.economy.stars_weight, &mut rng);
        child.economy.tech_weight = mutf(child.economy.tech_weight, &mut rng);
        child.economy.score_weight = mutf(child.economy.score_weight, &mut rng);
        child.economy.low_stars_threshold = mutf(child.economy.low_stars_threshold, &mut rng);

        // Mutate army genes
        child.army.base_weight = mutf(child.army.base_weight, &mut rng);
        child.army.hp_weight = mutf(child.army.hp_weight, &mut rng);
        child.army.status_weight = mutf(child.army.status_weight, &mut rng);
        child.army.defense_weight = mutf(child.army.defense_weight, &mut rng);
        child.army.loneliness_no_friends = mutf(child.army.loneliness_no_friends, &mut rng);

        // Mutate unit values
        for val in child.army.unit_values.values_mut() {
            *val = mutf(*val, &mut rng).clamp(0.01, 1.0);
        }

        // Mutate stage thresholds
        child.stages.early_threshold = mutf(child.stages.early_threshold, &mut rng).clamp(0.1, 0.5);
        child.stages.late_threshold = mutf(child.stages.late_threshold, &mut rng).clamp(0.5, 0.9);

        // Mutate exploration
        child.exploration.max_exploration_target = mutf(child.exploration.max_exploration_target, &mut rng).clamp(0.5, 1.0);

        // Mutate MCTS params
        child.mcts.exploration_constant = mutf(child.mcts.exploration_constant, &mut rng).clamp(0.1, 2.0);

        // Mutate Research genes
        child.research.org_fruit_multiplier = mutf(child.research.org_fruit_multiplier, &mut rng);
        child.research.hunting_game_multiplier = mutf(child.research.hunting_game_multiplier, &mut rng);
        child.research.fishing_fish_multiplier = mutf(child.research.fishing_fish_multiplier, &mut rng);
        child.research.farming_crop_multiplier = mutf(child.research.farming_crop_multiplier, &mut rng);
        child.research.mining_metal_multiplier = mutf(child.research.mining_metal_multiplier, &mut rng);
        child.research.forestry_forest_multiplier = mutf(child.research.forestry_forest_multiplier, &mut rng);
        child.research.climbing_mountain_multiplier = mutf(child.research.climbing_mountain_multiplier, &mut rng);
        child.research.sailing_water_multiplier = mutf(child.research.sailing_water_multiplier, &mut rng);
        child.research.navigation_ocean_multiplier = mutf(child.research.navigation_ocean_multiplier, &mut rng);
        child.research.riding_base = mutf(child.research.riding_base, &mut rng);
        child.research.riding_field_multiplier = mutf(child.research.riding_field_multiplier, &mut rng);
        child.research.archery_base = mutf(child.research.archery_base, &mut rng);
        child.research.strategy_base = mutf(child.research.strategy_base, &mut rng);
        child.research.chivalry_base = mutf(child.research.chivalry_base, &mut rng);
        child.research.smithery_base = mutf(child.research.smithery_base, &mut rng);
        child.research.roads_per_city_multiplier = mutf(child.research.roads_per_city_multiplier, &mut rng);
        child.research.trade_customs_multiplier = mutf(child.research.trade_customs_multiplier, &mut rng);
        child.research.philosophy_per_tech_multiplier = mutf(child.research.philosophy_per_tech_multiplier, &mut rng);
        child.research.diplomacy_per_player_multiplier = mutf(child.research.diplomacy_per_player_multiplier, &mut rng);
        child.research.tier_1_cost_offset = mutf(child.research.tier_1_cost_offset, &mut rng);
        child.research.tier_2_cost_offset = mutf(child.research.tier_2_cost_offset, &mut rng);
        child.research.tier_3_cost_offset = mutf(child.research.tier_3_cost_offset, &mut rng);

        child
    }

    /// Crossover two parent gene sets to produce a child.
    /// Each gene is randomly selected from one parent (uniform crossover).
    pub fn crossover(parent_a: &Self, parent_b: &Self) -> Self {
        let mut rng = rand::thread_rng();
        let mut child = parent_a.clone();

        // Helper: pick from a or b with 50% probability
        let pick = |a: f32, b: f32, rng: &mut rand::rngs::ThreadRng| -> f32 {
            if rng.r#gen::<bool>() { a } else { b }
        };

        // Crossover evaluator weights
        child.evaluator.early_perf_eco = pick(parent_a.evaluator.early_perf_eco, parent_b.evaluator.early_perf_eco, &mut rng);
        child.evaluator.early_perf_mil = pick(parent_a.evaluator.early_perf_mil, parent_b.evaluator.early_perf_mil, &mut rng);
        child.evaluator.early_perf_exp = pick(parent_a.evaluator.early_perf_exp, parent_b.evaluator.early_perf_exp, &mut rng);
        child.evaluator.early_perf_fow = pick(parent_a.evaluator.early_perf_fow, parent_b.evaluator.early_perf_fow, &mut rng);
        child.evaluator.early_dom_eco = pick(parent_a.evaluator.early_dom_eco, parent_b.evaluator.early_dom_eco, &mut rng);
        child.evaluator.early_dom_mil = pick(parent_a.evaluator.early_dom_mil, parent_b.evaluator.early_dom_mil, &mut rng);
        child.evaluator.early_dom_exp = pick(parent_a.evaluator.early_dom_exp, parent_b.evaluator.early_dom_exp, &mut rng);
        child.evaluator.early_dom_fow = pick(parent_a.evaluator.early_dom_fow, parent_b.evaluator.early_dom_fow, &mut rng);

        child.evaluator.mid_perf_eco = pick(parent_a.evaluator.mid_perf_eco, parent_b.evaluator.mid_perf_eco, &mut rng);
        child.evaluator.mid_perf_mil = pick(parent_a.evaluator.mid_perf_mil, parent_b.evaluator.mid_perf_mil, &mut rng);
        child.evaluator.mid_perf_exp = pick(parent_a.evaluator.mid_perf_exp, parent_b.evaluator.mid_perf_exp, &mut rng);
        child.evaluator.mid_perf_fow = pick(parent_a.evaluator.mid_perf_fow, parent_b.evaluator.mid_perf_fow, &mut rng);
        child.evaluator.mid_dom_eco = pick(parent_a.evaluator.mid_dom_eco, parent_b.evaluator.mid_dom_eco, &mut rng);
        child.evaluator.mid_dom_mil = pick(parent_a.evaluator.mid_dom_mil, parent_b.evaluator.mid_dom_mil, &mut rng);
        child.evaluator.mid_dom_exp = pick(parent_a.evaluator.mid_dom_exp, parent_b.evaluator.mid_dom_exp, &mut rng);
        child.evaluator.mid_dom_fow = pick(parent_a.evaluator.mid_dom_fow, parent_b.evaluator.mid_dom_fow, &mut rng);

        child.evaluator.end_perf_eco = pick(parent_a.evaluator.end_perf_eco, parent_b.evaluator.end_perf_eco, &mut rng);
        child.evaluator.end_perf_mil = pick(parent_a.evaluator.end_perf_mil, parent_b.evaluator.end_perf_mil, &mut rng);
        child.evaluator.end_perf_exp = pick(parent_a.evaluator.end_perf_exp, parent_b.evaluator.end_perf_exp, &mut rng);
        child.evaluator.end_perf_fow = pick(parent_a.evaluator.end_perf_fow, parent_b.evaluator.end_perf_fow, &mut rng);
        child.evaluator.end_dom_eco = pick(parent_a.evaluator.end_dom_eco, parent_b.evaluator.end_dom_eco, &mut rng);
        child.evaluator.end_dom_mil = pick(parent_a.evaluator.end_dom_mil, parent_b.evaluator.end_dom_mil, &mut rng);
        child.evaluator.end_dom_exp = pick(parent_a.evaluator.end_dom_exp, parent_b.evaluator.end_dom_exp, &mut rng);
        child.evaluator.end_dom_fow = pick(parent_a.evaluator.end_dom_fow, parent_b.evaluator.end_dom_fow, &mut rng);

        // Crossover ordering (pick random chunks from either parent)
        if rng.r#gen::<bool>() {
            child.ordering.capture_ruin = parent_b.ordering.capture_ruin;
            child.ordering.capture_village = parent_b.ordering.capture_village;
            child.ordering.capture_city = parent_b.ordering.capture_city;
        }
        if rng.r#gen::<bool>() {
            child.ordering.attack_kill = parent_b.ordering.attack_kill;
            child.ordering.attack_heavy_damage = parent_b.ordering.attack_heavy_damage;
            child.ordering.attack_light_damage = parent_b.ordering.attack_light_damage;
        }
        if rng.r#gen::<bool>() {
            child.ordering.summon_base = parent_b.ordering.summon_base;
            child.ordering.summon_threat_bonus = parent_b.ordering.summon_threat_bonus;
            child.ordering.summon_giant_bonus = parent_b.ordering.summon_giant_bonus;
        }
        if rng.r#gen::<bool>() {
            child.ordering.build_base = parent_b.ordering.build_base;
            child.ordering.adjacency_2_bonus = parent_b.ordering.adjacency_2_bonus;
            child.ordering.adjacency_3_bonus = parent_b.ordering.adjacency_3_bonus;
            child.ordering.adjacency_4plus_bonus = parent_b.ordering.adjacency_4plus_bonus;
        }

        // Crossover economy
        child.economy.income_weight = pick(parent_a.economy.income_weight, parent_b.economy.income_weight, &mut rng);
        child.economy.stars_weight = pick(parent_a.economy.stars_weight, parent_b.economy.stars_weight, &mut rng);
        child.economy.tech_weight = pick(parent_a.economy.tech_weight, parent_b.economy.tech_weight, &mut rng);
        child.economy.score_weight = pick(parent_a.economy.score_weight, parent_b.economy.score_weight, &mut rng);

        // Crossover unit values
        for (key, val) in &parent_b.army.unit_values {
            if rng.r#gen::<bool>() {
                child.army.unit_values.insert(key.clone(), *val);
            }
        }

        // Crossover stages
        child.stages.early_threshold = pick(parent_a.stages.early_threshold, parent_b.stages.early_threshold, &mut rng);
        child.stages.late_threshold = pick(parent_a.stages.late_threshold, parent_b.stages.late_threshold, &mut rng);

        // Crossover MCTS
        child.mcts.exploration_constant = pick(parent_a.mcts.exploration_constant, parent_b.mcts.exploration_constant, &mut rng);

        // Crossover Research
        if rng.r#gen::<bool>() {
            child.research = parent_b.research.clone();
        }

        child
    }

    /// Create a population of N gene variants from a seed (self), each with random mutation.
    pub fn spawn_population(&self, n: usize, mutation_rate: f32) -> Vec<Self> {
        let mut population = Vec::with_capacity(n);
        // First individual is the seed (unchanged)
        population.push(self.clone());
        // Rest are mutations
        for _ in 1..n {
            population.push(self.mutate(mutation_rate));
        }
        population
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_creation() {
        let genes = AIGenes::default();
        assert_eq!(genes.ordering.capture_ruin, 100.0);
        assert_eq!(genes.evaluator.early_dom_eco, 0.4);
        assert_eq!(genes.army.get_unit_value(UnitType::Giant), 0.80);
        assert_eq!(genes.army.get_unit_value(UnitType::Warrior), 0.38);
        assert_eq!(genes.stages.early_threshold, 0.3);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let genes = AIGenes::default();
        let json = serde_json::to_string_pretty(&genes).unwrap();
        let loaded: AIGenes = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.ordering.capture_ruin, genes.ordering.capture_ruin);
        assert_eq!(loaded.evaluator.early_dom_eco, genes.evaluator.early_dom_eco);
        assert_eq!(loaded.army.get_unit_value(UnitType::Giant), genes.army.get_unit_value(UnitType::Giant));
        assert_eq!(loaded.stages.early_threshold, genes.stages.early_threshold);
    }

    #[test]
    fn test_mutation_changes_values() {
        let genes = AIGenes::default();
        // Run multiple mutations — at least one should produce changes
        let mut any_changed = false;
        for _ in 0..20 {
            let mutated = genes.mutate(0.5);
            if mutated.ordering.capture_ruin != genes.ordering.capture_ruin
                || mutated.evaluator.early_dom_eco != genes.evaluator.early_dom_eco
                || mutated.army.base_weight != genes.army.base_weight
                || mutated.economy.income_weight != genes.economy.income_weight
                || mutated.ordering.attack_kill != genes.ordering.attack_kill
                || mutated.stages.early_threshold != genes.stages.early_threshold
            {
                any_changed = true;
                break;
            }
        }
        assert!(any_changed, "Mutation with high rate should change at least one gene across 20 attempts");
    }

    #[test]
    fn test_crossover_produces_valid() {
        let a = AIGenes::default();
        let b = a.mutate(0.3);
        let child = AIGenes::crossover(&a, &b);

        // Child should have valid values (no NaN, no infinity)
        assert!(child.ordering.capture_ruin.is_finite());
        assert!(child.evaluator.early_dom_eco.is_finite());
        assert!(!child.army.unit_values.is_empty());
    }

    #[test]
    fn test_spawn_population() {
        let seed = AIGenes::default();
        let pop = seed.spawn_population(10, 0.1);
        assert_eq!(pop.len(), 10);
        // First individual should be unchanged
        assert_eq!(pop[0].ordering.capture_ruin, seed.ordering.capture_ruin);
    }
}
