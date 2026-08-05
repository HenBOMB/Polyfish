use crate::moves::Move;
use crate::types::{
    AbilityType, CityRewardType, MoveType, RuinsRewardType, StructureType, TechnologyType, UnitType,
};
use serde::{Deserialize, Serialize};

use super::ReplayError;

/// Lossless engine-facing command identity. Source command ids are not part of
/// this representation; converters must resolve them before writing a replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase", deny_unknown_fields)]
pub enum ReplayCommand {
    Step {
        source: i32,
        target: i32,
    },
    Attack {
        source: i32,
        target: i32,
    },
    Capture {
        source: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reward: Option<RuinsRewardType>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revealed_tiles: Option<Vec<i32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        technology: Option<TechnologyType>,
    },
    Build {
        target: i32,
        structure: StructureType,
    },
    Research {
        technology: TechnologyType,
    },
    Summon {
        target: i32,
        unit: UnitType,
    },
    Upgrade {
        source: i32,
        unit: UnitType,
    },
    Ability {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target: Option<i32>,
        ability: AbilityType,
    },
    Reward {
        target: i32,
        reward: CityRewardType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revealed_tiles: Option<Vec<i32>>,
    },
    Harvest {
        target: i32,
    },
    EndTurn,
    Resign,
}

impl ReplayCommand {
    pub fn from_move(m: &dyn Move) -> Result<Self, ReplayError> {
        let component = |r: Result<usize, String>, name: &str| {
            r.map(|v| v as i32)
                .map_err(|message| ReplayError::CommandConversion {
                    move_summary: format!("{:?}", m),
                    message: format!("missing {name}: {message}"),
                })
        };
        Ok(match m.move_type() {
            MoveType::Step => Self::Step {
                source: component(m.source_idx(), "source")?,
                target: component(m.target_idx(), "target")?,
            },
            MoveType::Attack => Self::Attack {
                source: component(m.source_idx(), "source")?,
                target: component(m.target_idx(), "target")?,
            },
            MoveType::Capture => Self::Capture {
                source: component(m.source_idx(), "source")?,
                reward: None,
                revealed_tiles: None,
                technology: None,
            },
            MoveType::Build => Self::Build {
                target: component(m.target_idx(), "target")?,
                structure: m.structure_type().map_err(|message| {
                    ReplayError::CommandConversion {
                        move_summary: format!("{:?}", m),
                        message,
                    }
                })?,
            },
            MoveType::Research => Self::Research {
                technology: m
                    .tech_type()
                    .map_err(|message| ReplayError::CommandConversion {
                        move_summary: format!("{:?}", m),
                        message,
                    })?,
            },
            MoveType::Summon => {
                let unit = m
                    .unit_type()
                    .map_err(|message| ReplayError::CommandConversion {
                        move_summary: format!("{:?}", m),
                        message,
                    })?;
                let source = component(m.source_idx(), "source")?;
                if m.serialize()
                    .get("upgrade")
                    .and_then(|value| value.as_bool())
                    == Some(true)
                {
                    Self::Upgrade { source, unit }
                } else {
                    Self::Summon {
                        target: source,
                        unit,
                    }
                }
            }
            MoveType::Ability => Self::Ability {
                source: m.source_idx().ok().map(|v| v as i32),
                target: m.target_idx().ok().map(|v| v as i32),
                ability: m
                    .ability_type()
                    .map_err(|message| ReplayError::CommandConversion {
                        move_summary: format!("{:?}", m),
                        message,
                    })?,
            },
            MoveType::Reward => Self::Reward {
                target: component(m.target_idx(), "target")?,
                reward: m
                    .reward_type()
                    .map_err(|message| ReplayError::CommandConversion {
                        move_summary: format!("{:?}", m),
                        message,
                    })?,
                revealed_tiles: None,
            },
            MoveType::Harvest => Self::Harvest {
                target: component(m.target_idx(), "target")?,
            },
            MoveType::EndTurn => Self::EndTurn,
            MoveType::Resign => Self::Resign,
            MoveType::None => {
                return Err(ReplayError::CommandConversion {
                    move_summary: format!("{:?}", m),
                    message: "MoveType::None is not recordable".into(),
                });
            }
        })
    }

    pub fn matches_move(&self, m: &dyn Move) -> bool {
        let source = || m.source_idx().ok().map(|v| v as i32);
        let target = || m.target_idx().ok().map(|v| v as i32);
        match self {
            Self::Step {
                source: s,
                target: t,
            } => m.move_type() == MoveType::Step && source() == Some(*s) && target() == Some(*t),
            Self::Attack {
                source: s,
                target: t,
            } => m.move_type() == MoveType::Attack && source() == Some(*s) && target() == Some(*t),
            Self::Capture { source: s, .. } => {
                m.move_type() == MoveType::Capture && source() == Some(*s)
            }
            Self::Build {
                target: t,
                structure,
            } => {
                m.move_type() == MoveType::Build
                    && target() == Some(*t)
                    && m.structure_type().ok() == Some(*structure)
            }
            Self::Research { technology } => {
                m.move_type() == MoveType::Research && m.tech_type().ok() == Some(*technology)
            }
            Self::Summon { target: t, unit } => {
                m.move_type() == MoveType::Summon
                    && source() == Some(*t)
                    && m.unit_type().ok() == Some(*unit)
                    && m.serialize()
                        .get("upgrade")
                        .and_then(|value| value.as_bool())
                        != Some(true)
            }
            Self::Upgrade { source: s, unit } => {
                m.move_type() == MoveType::Summon
                    && source() == Some(*s)
                    && m.unit_type().ok() == Some(*unit)
                    && m.serialize()
                        .get("upgrade")
                        .and_then(|value| value.as_bool())
                        == Some(true)
            }
            Self::Ability {
                source: s,
                target: t,
                ability,
            } => {
                m.move_type() == MoveType::Ability
                    && m.ability_type().ok() == Some(*ability)
                    && source() == *s
                    && target() == *t
            }
            Self::Reward {
                target: t, reward, ..
            } => {
                m.move_type() == MoveType::Reward
                    && target() == Some(*t)
                    && m.reward_type().ok() == Some(*reward)
            }
            Self::Harvest { target: t } => {
                m.move_type() == MoveType::Harvest && target() == Some(*t)
            }
            Self::EndTurn => m.move_type() == MoveType::EndTurn,
            Self::Resign => m.move_type() == MoveType::Resign,
        }
    }

    pub fn tile_indices(&self) -> impl Iterator<Item = i32> + '_ {
        let (a, b, extra): (Option<i32>, Option<i32>, Option<&[i32]>) = match self {
            Self::Step { source, target } | Self::Attack { source, target } => {
                (Some(*source), Some(*target), None)
            }
            Self::Capture {
                source,
                revealed_tiles,
                ..
            } => (Some(*source), None, revealed_tiles.as_deref()),
            Self::Build { target, .. }
            | Self::Summon { target, .. }
            | Self::Reward { target, .. }
            | Self::Harvest { target } => (
                Some(*target),
                None,
                match self {
                    Self::Reward { revealed_tiles, .. } => revealed_tiles.as_deref(),
                    _ => None,
                },
            ),
            Self::Ability { source, target, .. } => (*source, *target, None),
            Self::Upgrade { source, .. } => (Some(*source), None, None),
            Self::Research { .. } | Self::EndTurn | Self::Resign => (None, None, None),
        };
        a.into_iter()
            .chain(b)
            .chain(extra.into_iter().flatten().copied())
    }
}
