use crate::states::CityState;

pub const CITY_BASE_SCORE: i32 = 100;
pub const CITY_LEVEL_UP_SCORE: i32 = 50;
pub const CITY_POPULATION_SCORE: i32 = 5;
pub const CITY_TERRITORY_SCORE: i32 = 20;

pub fn get_city_score(city: &CityState) -> i32 {
    CITY_BASE_SCORE
        + (city.level - 1) * CITY_LEVEL_UP_SCORE
        + city.population * CITY_POPULATION_SCORE
}
