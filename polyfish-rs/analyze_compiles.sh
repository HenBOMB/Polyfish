#!/usr/bin/env bash
# Usage: ./analyze_compiles.sh compile.log
# Summarizes MPSGraph executable-compile cost from self_play stderr
# (the METAL_COMPILE lines emitted by metal_network.rs::build_and_compile).
log="${1:-compile.log}"

grep METAL_COMPILE "$log" | awk '
{
  delete v
  for (i=1;i<=NF;i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
  net=v["net"]; size=v["size"]; ms=v["ms"]
  total_compiles++; total_ms += ms
  nets[net]=1; sizes[size]++
  net_compiles[net]=v["net_total_compiles"]; net_ms[net]=v["net_total_ms"]
}
END {
  nworkers=0; for (n in nets) nworkers++
  ndistinct=0; for (s in sizes) ndistinct++
  printf "workers (distinct nets):   %d\n", nworkers
  printf "total compiles (all nets): %d\n", total_compiles
  printf "total compile time:        %.1f ms  (%.2f s)\n", total_ms, total_ms/1000
  printf "distinct batch sizes:      %d\n", ndistinct
  print  "--- per-worker totals ---"
  for (n in nets) printf "  %s  compiles=%s  ms=%s\n", n, net_compiles[n], net_ms[n]
}'

echo "--- compiles per batch size (count size, desc) ---"
grep METAL_COMPILE "$log" \
  | sed -n 's/.* size=\([0-9]*\) .*/\1/p' \
  | sort -n | uniq -c | sort -rn | head -20
