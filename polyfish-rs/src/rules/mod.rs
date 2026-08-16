//! Single source of truth for derived game quantities.
//!
//! Every rule here is expressed in terms of `settings/*` (the data SSOT) and the
//! engine's own execution paths. Consumers — the AI, the evaluator, the binaries
//! and the engine itself — call these instead of re-deriving, because six bugs in
//! one session all traced to the same rule being implemented in several places
//! and the copies drifting apart.
//!
//! ## Shape
//!
//! Functions come in two tiers:
//!
//! * `foo_with(...)` — the core. Takes already-resolved context, performs no
//!   lookups and no allocation. The rollout path calls this.
//! * `foo(...)` — convenience. Resolves the context, then delegates.
//!
//! Both share one body, so they cannot disagree. A parameter may control **cost,
//! never the answer**: anything that would change the result is a differently
//! named function (see `partner_count` vs `partner_count_planned` vs
//! `partner_ceiling`), never a flag.

pub mod capture;
pub mod combat;
pub mod eco_plan;
pub mod economy;
pub mod vision;
