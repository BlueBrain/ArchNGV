#!/usr/bin/env bash

set -euo pipefail

EXPECTED="$1"
ACTUAL="$2"

function h5_compare() {
    local relpath="$1"
    echo "Verifying $relpath..."
    h5diff -c "$EXPECTED/$relpath" "$ACTUAL/$relpath"
}

function h5_compare_all() {
    local dirname="$1"
    h5_files=$(find "$EXPECTED/$dirname" -maxdepth 1 -name "*.h5" -printf '%f\n' | sort)
    for fname in $h5_files; do
        h5_compare "$dirname/$fname"
    done
}

h5_compare_all "."
h5_compare_all "./microdomains"
h5_compare_all "./morphologies"

echo "OK!"
