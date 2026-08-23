//! Unknown enum ids arriving from JSON captures or the HTTP API must map to a
//! defined variant, not to an invalid discriminant (issue #49).

use polyfish::states::TribeState;
use polyfish::types::{AbilityType, CityRewardType, StructureType, TechnologyType, UnitType};

fn tribe_with_tech(ids: &str) -> TribeState {
    serde_json::from_str(&format!(r#"{{"id": 1, "type": 2, "tech_vanilla": {ids}}}"#))
        .expect("tribe deserializes")
}

#[test]
fn tech_list_maps_unknown_ids_to_basic() {
    // 11, 50 and 200 are gaps / out of range in the sparse TechnologyType repr.
    let tribe = tribe_with_tech("[1, 11, 50, 200, -7]");
    let got: Vec<TechnologyType> = tribe.tech_vanilla.iter().map(|t| t.tech_type).collect();
    assert_eq!(
        got,
        vec![
            TechnologyType::Riding,
            TechnologyType::Basic,
            TechnologyType::Basic,
            TechnologyType::Basic,
            TechnologyType::Basic,
        ]
    );
    assert!(tribe.tech_vanilla.iter().all(|t| t.discovered));
}

#[test]
fn tech_list_still_accepts_full_state_entries() {
    let tribe: TribeState = serde_json::from_str(
        r#"{"id": 1, "type": 2, "tech_vanilla": [{"type": 2, "discovered": true, "discoveredTurn": 4}]}"#,
    )
    .expect("tribe deserializes");
    assert_eq!(tribe.tech_vanilla.len(), 1);
    assert_eq!(tribe.tech_vanilla[0].discovered_turn, 4);
}

#[test]
fn http_id_conversions_fall_back_to_none() {
    for bad in [99, 250, -3, 1_000_000] {
        assert_eq!(AbilityType::from(bad), AbilityType::None);
        assert_eq!(StructureType::from(bad), StructureType::None);
        assert_eq!(UnitType::from(bad), UnitType::None);
        assert_eq!(CityRewardType::from(bad), CityRewardType::None);
        assert_eq!(TechnologyType::from(bad), TechnologyType::Basic);
    }

    assert_eq!(AbilityType::from(7), AbilityType::Recover);
    assert_eq!(StructureType::from(21), StructureType::Mine);
}
