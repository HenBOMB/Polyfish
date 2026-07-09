CREATE TABLE training_metrics (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at timestamptz DEFAULT now(),
  run_id bigint,
  iter_started_at timestamptz,
  iteration int,
  timestamp float,
  games_file text,
  avg_score float,
  max_score float,
  p1_avg float,
  p2_avg float,
  loss float,
  policy_loss float,
  value_loss float,
  avg_captures float,
  avg_cap_ruins float,
  avg_cap_villages float,
  avg_cap_cities float,
  avg_cap_capitals float,
  avg_harvests float,
  avg_builds float,
  avg_research float,
  avg_attacks float,
  avg_moves float,
  match_type text
);

-- Turn on Realtime for the table so the Telegram Agent receives events
ALTER PUBLICATION supabase_realtime ADD TABLE training_metrics;
