//! Turning finished games into training rows.
//!
//! `ShardBuffers` holds one shard's worth of per-step targets. `push_game`
//! computes a game's TD(lambda) value label and every aux-head ground truth
//! (see `labels`) and appends one row per decision; `maybe_flush` / `finish`
//! drain the buffers through `shard::flush_shard`.

use candle_core::{Device, Tensor};
use polyfish::ai::features;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use strum::IntoEnumIterator;

use crate::cli::Args;
use crate::labels::{ArmyStep, CitySptStep, FINAL_OUTCOME_REL_W, GOOD_BOT_FINAL_SCORE,
                    LabelStep, SptStep, TerritoryStep, army_checkpoints_by_player,
                    army_target, city_spt_checkpoints, city_spt_target, macro_policy_targets,
                    ownership_from_pov, siege_pressure_target, spt_checkpoints_by_player,
                    spt_target, td_lambda_labels, territory_checkpoints_by_player,
                    territory_target};
use crate::result::{GameResult, HistoryStep};
use crate::shard::{SHARD_GAMES, flush_shard};
use crate::stats::is_net_seat;

pub(crate) struct ShardBuffers {
    // Aggregate results
    collected_spatial_maps: Vec<Tensor>,
    collected_player_states: Vec<Tensor>,

    // Decomposed policy targets (7 heads)
    collected_action_type: Vec<Vec<f32>>,
    collected_source_spatial: Vec<Vec<f32>>,
    collected_target_spatial: Vec<Vec<f32>>,
    collected_option: Vec<Vec<f32>>,

    collected_values: Vec<f32>,
    collected_progress: Vec<f32>,

    // Aux-head targets (see the aux_* helpers above GameResult).
    num_techs: usize,
    collected_aux_own: Vec<Vec<f32>>,
    collected_aux_fog: Vec<Vec<f32>>,
    collected_aux_spt: Vec<f32>, // flat, 2 per step
    // Horizon-compression Stage 1b (EXP_ELO_120): territory tile count now
    // vs. turn+5, same flat-2-per-step shape as aux_spt. Monotone "reached",
    // not "held" — see labels.rs's TerritoryStep doc.
    collected_aux_territory5: Vec<f32>,
    // Horizon-compression Stage 1a (EXP_ELO_120): row-masked like
    // macro_ballot/macro_mask, not the file-level AUX_DIMS convention --
    // presence varies per-row (once per (turn, pov), same shape as
    // macro_ballot), not per-file. The mask rides train.py's existing
    // generic aux_mask[k] plumbing (already a per-sample tensor), so no
    // compute_loss changes are needed -- only the shard-loading side reads
    // this mask instead of the blanket per-file one.
    collected_aux_eco_ceiling: Vec<Vec<f32>>,
    collected_eco_ceiling_mask: Vec<f32>,
    // Horizon-compression Stage 2 (EXP_ELO_120): plain per-file AUX_DIMS
    // convention, no row mask needed -- unlike eco_ceiling, siege_opens is
    // always computable post-game, for every row.
    collected_aux_pressure: Vec<f32>,
    // Horizon-compression Stage 3 (EXP_ELO_120): army value differential,
    // third copy of the SPT+5 flat-2-per-step shape.
    collected_aux_army5: Vec<f32>,
    collected_aux_tech: Vec<Vec<f32>>,
    collected_aux_pursuit: Vec<f32>, // scalar per step
    collected_aux_city_spt: Vec<Vec<f32>>, // board-sized per step

    // EXP_ELO_061 (Stage 3b): macro policy targets. Per-ROW mask, not just
    // per-file — even a macro-mcts-heavy run has steps with no ballot (the
    // opponent seat, an anchor game). Zero-filled + mask=0 there, matching
    // the aux-head-per-key-mask lesson: never let an absent target train
    // toward a fake zero.
    collected_macro_stance: Vec<Vec<f32>>,
    collected_macro_order: Vec<Vec<f32>>,
    collected_macro_mask: Vec<f32>,
    shard_files: Vec<String>,
    games_in_shard: usize,
    /// `trace_games` for --trace-villages runs, else `games`: diagnostic
    /// runs must not match the training loop's `games_*` glob.
    shard_prefix: &'static str,
    run_ts: u64,
    value_calib_file: Option<File>,
}

impl ShardBuffers {
    pub(crate) fn new(args: &Args, run_ts: u64) -> Self {
        Self {
            collected_spatial_maps: Vec::new(),
            collected_player_states: Vec::new(),
            collected_action_type: Vec::new(),
            collected_source_spatial: Vec::new(),
            collected_target_spatial: Vec::new(),
            collected_option: Vec::new(),
            collected_values: Vec::new(),
            collected_progress: Vec::new(),
            num_techs: polyfish::types::TechnologyType::iter().count(),
            collected_aux_own: Vec::new(),
            collected_aux_fog: Vec::new(),
            collected_aux_spt: Vec::new(),
            collected_aux_territory5: Vec::new(),
            collected_aux_eco_ceiling: Vec::new(),
            collected_eco_ceiling_mask: Vec::new(),
            collected_aux_pressure: Vec::new(),
            collected_aux_army5: Vec::new(),
            collected_aux_tech: Vec::new(),
            collected_aux_pursuit: Vec::new(),
            collected_aux_city_spt: Vec::new(),
            collected_macro_stance: Vec::new(),
            collected_macro_order: Vec::new(),
            collected_macro_mask: Vec::new(),
            shard_files: Vec::new(),
            games_in_shard: 0usize,
            shard_prefix: if args.trace_villages { "trace_games" } else { "games" },
            run_ts,
            value_calib_file: args
                .dump_value_calib
                .as_ref()
                .and_then(|p| File::create(p).ok()),
        }
    }

    /// Appends one row per decision in `result`, carrying its value label
    /// and every aux-head target. Consumes the game's history.
    pub(crate) fn push_game(&mut self, result: GameResult, args: &Args) {
        let num_techs = self.num_techs;
        let Self {
            collected_spatial_maps,
            collected_player_states,
            collected_action_type,
            collected_source_spatial,
            collected_target_spatial,
            collected_option,
            collected_values,
            collected_progress,
            collected_aux_own,
            collected_aux_fog,
            collected_aux_spt,
            collected_aux_territory5,
            collected_aux_eco_ceiling,
            collected_eco_ceiling_mask,
            collected_aux_pressure,
            collected_aux_army5,
            collected_aux_tech,
            collected_aux_pursuit,
            collected_aux_city_spt,
            collected_macro_stance,
            collected_macro_order,
            collected_macro_mask,
            value_calib_file, ..
        } = self;
        let final_scores = &result.scores;

        let label_steps: Vec<LabelStep> = result.history.iter().map(LabelStep::from).collect();
        // EXP_ELO_025: outcome-space labels — z anchors the TD tail too.
        let wl_z: Option<HashMap<i32, f32>> = if args.wl_labels {
            Some(
                result
                    .scores
                    .keys()
                    .map(|&id| (id, if id == result.winner_id { 1.0 } else { -1.0 }))
                    .collect(),
            )
        } else {
            None
        };
        let td_deltas = td_lambda_labels(
            &label_steps,
            &result.final_potentials,
            args.td_lambda,
            args.label_rel_w,
            wl_z.as_ref(),
            args.td_missing_bootstrap,
        );

        let spt_steps: Vec<SptStep> = result
            .history
            .iter()
            .map(|s| SptStep {
                player_id: s.player_id,
                turn: s.turn,
                my_spt: s.my_spt,
                opp_spt: s.opp_spt,
            })
            .collect();
        let spt_cp = spt_checkpoints_by_player(&spt_steps);
        let territory_steps: Vec<TerritoryStep> = result
            .history
            .iter()
            .map(|s| TerritoryStep {
                player_id: s.player_id,
                turn: s.turn,
                my_territory: s.my_territory,
                opp_territory: s.opp_territory,
            })
            .collect();
        let territory_cp = territory_checkpoints_by_player(&territory_steps);
        let army_steps: Vec<ArmyStep> = result
            .history
            .iter()
            .map(|s| ArmyStep {
                player_id: s.player_id,
                turn: s.turn,
                my_army: s.my_army,
                opp_army: s.opp_army,
            })
            .collect();
        let army_cp = army_checkpoints_by_player(&army_steps);
        let city_spt_steps: Vec<CitySptStep> = result
            .history
            .iter()
            .map(|s| CitySptStep {
                player_id: s.player_id,
                turn: s.turn,
                cities: s.city_spt.clone(),
            })
            .collect();
        let city_spt_cp = city_spt_checkpoints(&city_spt_steps);

        let game_winner_id = result.winner_id;

        for (step_idx, step) in result.history.into_iter().enumerate() {
            let HistoryStep {
                features,
                policy: policy_data,
                player_id: p_id,
                turn,
                enemy_units,
                pursuit,
                my_score: step_my_score,
                opp_score: step_opp_score,
                root_value: step_root_value,
                root_own_value: step_root_own_value,
                macro_ballot,
                eco_ceiling,
                ..
            } = step;
            let flat_map = features
                .spatial_map
                .flatten_all()
                .expect("BUG: Failed to flatten spatial map tensor");
            collected_spatial_maps.push(flat_map);

            let flat_player = features
                .player_state
                .flatten_all()
                .expect("BUG: Failed to flatten player state tensor");
            collected_player_states.push(flat_player);

            collected_action_type.push(policy_data.action_type);
            collected_source_spatial.push(policy_data.source_spatial);
            collected_target_spatial.push(policy_data.target_spatial);
            collected_option.push(policy_data.move_option);

            // Perfection: Score-based value target
            // Every game produces a meaningful score — use normalized differential
            let my_final = final_scores.get(&p_id).copied().unwrap_or(0) as f32;
            let opp_final = final_scores
                .iter()
                .filter(|(id, _)| **id != p_id)
                .map(|(_, score)| *score as f32)
                .next()
                .unwrap_or(0.0);

            // Asymmetric Reward Shaping to fix P1 advantage
            let (mut my_adjusted, mut opp_adjusted) = (my_final, opp_final);
            if !args.no_reward_shaping {
                let penalty = 0.05; // 5% adjustment
                if p_id == 1 {
                    my_adjusted = my_final * (1.0 - penalty);
                    opp_adjusted = opp_final * (1.0 + penalty);
                } else if p_id == 2 {
                    my_adjusted = my_final * (1.0 + penalty);
                    opp_adjusted = opp_final * (1.0 - penalty);
                }
            }

            // Normalize by combined economic activity with scaling multiplier
            let combined_score = my_adjusted + opp_adjusted;
            // to spread distribution into useful training range
            let scaling_factor = args.outcome_scale;
            let relative_outcome = if combined_score > 0.0 {
                let ratio = (my_adjusted - opp_adjusted) / combined_score;
                (ratio * scaling_factor).clamp(-1.0, 1.0)
            } else {
                0.0 // Both players scored 0 - treat as draw
            };

            // Absolute value: final score vs fixed yardstick, not current scoreboard.
            let abs_outcome = (my_final / GOOD_BOT_FINAL_SCORE).clamp(0.0, 1.0) * 2.0 - 1.0;
            let final_outcome = if args.wl_labels {
                // EXP_ELO_011: the score ratio under-punishes close losses
                // (4257 vs 4676 reads -0.14); win/loss makes them -1.
                if p_id == game_winner_id {
                    1.0
                } else {
                    -1.0
                }
            } else {
                (FINAL_OUTCOME_REL_W * relative_outcome
                    + (1.0 - FINAL_OUTCOME_REL_W) * abs_outcome)
                    .clamp(-1.0, 1.0)
            };

            let value = if !args.no_reward_shaping {
                // TD delta carries per-action credit; the final-outcome tail
                // carries the long-horizon signal.
                (args.td_w * td_deltas[step_idx] + (1.0 - args.td_w) * final_outcome)
                    .clamp(-1.0, 1.0)
            } else {
                final_outcome.clamp(-1.0, 1.0)
            };

            collected_values.push(value);

            // Value-head calibration: NN prediction vs current-score-ratio vs
            // the actual final outcome, for net seats that ran a real search.
            if let (Some(f), Some(rv)) = (value_calib_file.as_mut(), step_root_value) {
                if is_net_seat(result.roles, p_id) {
                    let raw = step_root_own_value.unwrap_or(rv);
                    let _ = writeln!(
                        f,
                        "{{\"turn\":{turn},\"my\":{step_my_score},\"opp\":{step_opp_score},\"root_value\":{rv},\"raw_value\":{raw},\"final_outcome\":{final_outcome},\"value_target\":{value}}}"
                    );
                }
            }

            let my_final_cities = result.final_cities.get(&p_id).copied().unwrap_or(0) as f32;
            let total_cities = result.total_cities as f32;
            let progress_target = if total_cities > 0.0 {
                (my_final_cities / total_cities).clamp(0.0, 1.0) * 2.0 - 1.0
            } else {
                -1.0
            };
            collected_progress.push(progress_target);

            let opp_id = final_scores
                .keys()
                .copied()
                .find(|id| *id != p_id)
                .unwrap_or(p_id);
            collected_aux_own.push(ownership_from_pov(&result.final_owner, p_id));
            collected_aux_fog.push(enemy_units);
            let (spt_my, spt_opp) = spt_target(
                spt_cp.get(&p_id),
                turn,
                result.final_spt.get(&p_id).copied().unwrap_or(0),
                result.final_spt.get(&opp_id).copied().unwrap_or(0),
            );
            collected_aux_spt.push(spt_my as f32 / 20.0);
            collected_aux_spt.push(spt_opp as f32 / 20.0);
            // EXP_ELO_120: territory tile counts run higher than SPT (a
            // multi-city empire can hold 60+ tiles on an 11x11 map) --
            // deliberately a different normalization constant, not a reuse
            // of SPT's /20.0.
            let (terr_my, terr_opp) = territory_target(
                territory_cp.get(&p_id),
                turn,
                result.final_territory.get(&p_id).copied().unwrap_or(0),
                result.final_territory.get(&opp_id).copied().unwrap_or(0),
            );
            collected_aux_territory5.push(terr_my as f32 / 40.0);
            collected_aux_territory5.push(terr_opp as f32 / 40.0);
            collected_aux_pursuit.push(pursuit);
            if let Some((candidates, visits)) = &macro_ballot {
                let (stance, order) = macro_policy_targets(candidates, visits);
                collected_macro_stance.push(stance);
                collected_macro_order.push(order);
                collected_macro_mask.push(1.0);
            } else {
                collected_macro_stance.push(vec![0.0; 4]);
                collected_macro_order.push(vec![0.0; 3 * features::MAP_SIZE * features::MAP_SIZE]);
                collected_macro_mask.push(0.0);
            }
            if let Some(ceiling) = eco_ceiling {
                collected_aux_eco_ceiling.push(ceiling.to_vec());
                collected_eco_ceiling_mask.push(1.0);
            } else {
                collected_aux_eco_ceiling.push(vec![0.0; 4]);
                collected_eco_ceiling_mask.push(0.0);
            }
            collected_aux_pressure.push(siege_pressure_target(&result.siege_opens, turn, opp_id));
            let (army_my, army_opp) = army_target(
                army_cp.get(&p_id),
                turn,
                result.final_army.get(&p_id).copied().unwrap_or(0.0),
                result.final_army.get(&opp_id).copied().unwrap_or(0.0),
            );
            collected_aux_army5.push(army_my);
            collected_aux_army5.push(army_opp);
            collected_aux_city_spt.push(city_spt_target(
                city_spt_cp.get(&p_id),
                turn,
                features::MAP_SIZE * features::MAP_SIZE,
            ));
            collected_aux_tech.push(
                result
                    .final_tech
                    .get(&opp_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; num_techs]),
            );
        }
    }

    /// Writes a shard once SHARD_GAMES games have accumulated.
    pub(crate) fn maybe_flush(&mut self, device: &Device) -> anyhow::Result<()> {
        let (shard_prefix, run_ts) = (self.shard_prefix, self.run_ts);
    self.games_in_shard += 1;
    if self.games_in_shard >= SHARD_GAMES && !self.collected_spatial_maps.is_empty() {
        let path = format!(
            "{shard_prefix}_{run_ts}_p{}.safetensors",
            self.shard_files.len()
        );
        flush_shard(
            std::mem::take(&mut self.collected_spatial_maps),
            std::mem::take(&mut self.collected_player_states),
            std::mem::take(&mut self.collected_action_type),
            std::mem::take(&mut self.collected_source_spatial),
            std::mem::take(&mut self.collected_target_spatial),
            std::mem::take(&mut self.collected_option),
            std::mem::take(&mut self.collected_values),
            std::mem::take(&mut self.collected_progress),
            std::mem::take(&mut self.collected_aux_own),
            std::mem::take(&mut self.collected_aux_fog),
            std::mem::take(&mut self.collected_aux_spt),
            std::mem::take(&mut self.collected_aux_territory5),
            std::mem::take(&mut self.collected_aux_eco_ceiling),
            std::mem::take(&mut self.collected_eco_ceiling_mask),
            std::mem::take(&mut self.collected_aux_pressure),
            std::mem::take(&mut self.collected_aux_army5),
            std::mem::take(&mut self.collected_aux_pursuit),
            std::mem::take(&mut self.collected_aux_city_spt),
            std::mem::take(&mut self.collected_aux_tech),
            self.num_techs,
            std::mem::take(&mut self.collected_macro_stance),
            std::mem::take(&mut self.collected_macro_order),
            std::mem::take(&mut self.collected_macro_mask),
            device,
            &path,
        )?;
        self.shard_files.push(path);
        self.games_in_shard = 0;
    }
        Ok(())
    }

    /// Writes the trailing partial shard and returns every file written.
    pub(crate) fn finish(&mut self, device: &Device) -> anyhow::Result<Vec<String>> {
        let (shard_prefix, run_ts) = (self.shard_prefix, self.run_ts);
    // Final partial shard.
    if !self.collected_spatial_maps.is_empty() {
        let path = format!(
            "{shard_prefix}_{run_ts}_p{}.safetensors",
            self.shard_files.len()
        );
        flush_shard(
            std::mem::take(&mut self.collected_spatial_maps),
            std::mem::take(&mut self.collected_player_states),
            std::mem::take(&mut self.collected_action_type),
            std::mem::take(&mut self.collected_source_spatial),
            std::mem::take(&mut self.collected_target_spatial),
            std::mem::take(&mut self.collected_option),
            std::mem::take(&mut self.collected_values),
            std::mem::take(&mut self.collected_progress),
            std::mem::take(&mut self.collected_aux_own),
            std::mem::take(&mut self.collected_aux_fog),
            std::mem::take(&mut self.collected_aux_spt),
            std::mem::take(&mut self.collected_aux_territory5),
            std::mem::take(&mut self.collected_aux_eco_ceiling),
            std::mem::take(&mut self.collected_eco_ceiling_mask),
            std::mem::take(&mut self.collected_aux_pressure),
            std::mem::take(&mut self.collected_aux_army5),
            std::mem::take(&mut self.collected_aux_pursuit),
            std::mem::take(&mut self.collected_aux_city_spt),
            std::mem::take(&mut self.collected_aux_tech),
            self.num_techs,
            std::mem::take(&mut self.collected_macro_stance),
            std::mem::take(&mut self.collected_macro_order),
            std::mem::take(&mut self.collected_macro_mask),
            device,
            &path,
        )?;
        self.shard_files.push(path);
    }
        Ok(std::mem::take(&mut self.shard_files))
    }
}
