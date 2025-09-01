#!/usr/bin/env bash
set -euo pipefail
root="/lustre/scratch/dlca"
mkdir -p "$root"/checkpoint "$root"/inputs "$root"/prediction "$root"/logs
echo "DLCA syncpoints created under $root"

