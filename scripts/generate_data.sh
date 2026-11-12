#!/usr/bin/env bash
# Generate the Synthea source data.
#
#   scripts/generate_data.sh            full population into data/raw and the sample into data/sample
#   scripts/generate_data.sh full       full population only
#   scripts/generate_data.sh sample     sample only
#
# Environment overrides:
#   POPULATION   living patients in the full run (default 12000)
#   SAMPLE_POP   living patients in the sample (default 150)
#   JAVA_HEAP    JVM heap (default 3g)
#
# The output is deterministic: fixed seed, clinician seed, reference date and end date give
# the same rows every run. Row order inside a CSV can differ because Synthea writes from
# several threads, and the summary columns of payers.csv can differ in the last decimal.
# The warehouse depends on neither. The committed sample is sorted so its files are stable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"
POPULATION="${POPULATION:-12000}"
SAMPLE_POP="${SAMPLE_POP:-150}"
JAVA_HEAP="${JAVA_HEAP:-3g}"

JAR_DIR="$ROOT/data/raw/_jar"
JAR="$JAR_DIR/synthea-with-dependencies.jar"
JAR_URL="https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"

# Only the files the warehouse reads. claims.csv and claims_transactions.csv are left out
# on purpose: they are about two thirds of the export by size and the warehouse takes
# cost from the clinical files (see warehouse/README.md).
FILES="patients.csv,encounters.csv,conditions.csv,medications.csv,procedures.csv,observations.csv,immunizations.csv,payer_transitions.csv,payers.csv"

REFERENCE_DATE=20251231
END_DATE=20251231
YEARS_OF_HISTORY=15
STATE=Pennsylvania

mkdir -p "$JAR_DIR"
if [ ! -s "$JAR" ]; then
  echo "Downloading Synthea jar"
  curl -fsSL -o "$JAR.part" "$JAR_URL"
  mv "$JAR.part" "$JAR"
fi

run_synthea() {
  local pop="$1" seed="$2" out="$3"
  local work="$ROOT/data/raw/_work_$seed"
  rm -rf "$work"
  mkdir -p "$work" "$out"
  local start
  start=$(date +%s)
  (
    cd "$work"
    java "-Xmx$JAVA_HEAP" -jar "$JAR" \
      -p "$pop" -s "$seed" -cs "$seed" \
      -r "$REFERENCE_DATE" -e "$END_DATE" \
      --exporter.csv.export=true \
      --exporter.csv.included_files="$FILES" \
      --exporter.fhir.export=false \
      --exporter.hospital.fhir.export=false \
      --exporter.practitioner.fhir.export=false \
      --exporter.years_of_history="$YEARS_OF_HISTORY" \
      --exporter.baseDirectory="$work/output/" \
      "$STATE" > "$work/synthea.log" 2>&1
  )
  local secs=$(( $(date +%s) - start ))
  rm -f "$out"/*.csv "$out"/*.csv.gz
  mv "$work"/output/csv/*.csv "$out"/
  grep -E "^Records: " "$work/synthea.log" | tail -1 > "$out/_generation.txt" || true
  {
    echo "population_requested=$pop"
    echo "seed=$seed"
    echo "clinician_seed=$seed"
    echo "reference_date=$REFERENCE_DATE"
    echo "end_date=$END_DATE"
    echo "years_of_history=$YEARS_OF_HISTORY"
    echo "state=$STATE"
    echo "wall_seconds=$secs"
    echo "synthea_jar_sha256=$(sha256sum "$JAR" | cut -d' ' -f1)"
    echo "synthea_version=$(unzip -p "$JAR" META-INF/MANIFEST.MF 2>/dev/null | grep -i '^Build-Version' | cut -d' ' -f2 | tr -d '\r')"
    echo "synthea_build_timestamp=$(unzip -p "$JAR" META-INF/MANIFEST.MF 2>/dev/null | grep -i '^Build-Timestamp' | cut -d' ' -f2 | tr -d '\r')"
  } >> "$out/_generation.txt"
  rm -rf "$work"
  echo "Generated $pop living patients into $out in ${secs}s"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "full" ]; then
  run_synthea "$POPULATION" 42 "$ROOT/data/raw"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "sample" ]; then
  run_synthea "$SAMPLE_POP" 7 "$ROOT/data/sample"
  # The sample is committed, so sort the rows to keep the files stable between runs.
  for f in "$ROOT"/data/sample/*.csv; do
    { head -n 1 "$f"; tail -n +2 "$f" | LC_ALL=C sort; } > "$f.sorted"
    mv "$f.sorted" "$f"
  done
  gzip -9 -n -f "$ROOT"/data/sample/*.csv
  echo "Sample size: $(du -sh "$ROOT/data/sample" | cut -f1)"
fi
