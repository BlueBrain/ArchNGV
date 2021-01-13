#!/usr/bin/env bash

set -euo pipefail

EXPECTED="$1"
ACTUAL="$2"

function h5_compare() {
    local relpath="$1"
    echo Verifying: h5diff -v -c --delta=5e-07 "$EXPECTED/$relpath" "$ACTUAL/$relpath"
    h5diff -v -c --delta=5e-07 "$EXPECTED/$relpath" "$ACTUAL/$relpath"
}

function h5_compare_all() {
    local dirname="$1"
    h5_files=$(find "$EXPECTED/$dirname" -maxdepth 1 -name "*.h5" -printf '%f\n' | sort)
    for fname in $h5_files; do
        h5_compare "$dirname/$fname"
    done
}

function morph_compare() {
    local relpath="$1"
    echo "Verifying $relpath..."
    morph-tool diff "$EXPECTED/$relpath" "$ACTUAL/$relpath"
}

function morph_compare_all() {
    local dirname="$1"
    h5_files=$(find "$EXPECTED/$dirname" -maxdepth 1 -name "*.h5" -printf '%f\n' | sort)
    for fname in $h5_files; do
        if [[ $fname =~ 'annotation' ]]; then
           h5_compare "$dirname/$fname"
        else
           morph_compare "$dirname/$fname"
        fi
    done
}

h5_compare_all "."
h5_compare_all "./microdomains"
h5_compare_all "./sonata/nodes"
h5_compare_all "./sonata/edges"

morph_compare_all "./morphologies"

echo "OK!"