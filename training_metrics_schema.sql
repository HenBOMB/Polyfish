CREATE TABLE training_metrics (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at timestamptz DEFAULT now(),
  iteration int,
  timestamp float,
  avg_score float,
  max_score float,
  p1_avg float,
  p2_avg float,
  loss float,
  avg_captures float,
  avg_harvests float,
  avg_builds float,
  avg_research float,
  avg_attacks float
);

-- Turn on Realtime for the table so the Telegram Agent receives events
ALTER PUBLICATION supabase_realtime ADD TABLE training_metrics;
