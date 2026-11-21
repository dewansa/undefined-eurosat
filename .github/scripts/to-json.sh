#!/usr/bin/env bash
# Turn names on stdin (one per line) into a JSON list for a job matrix: ["a","b"].
jq -R 'select(length > 0)' | jq -sc .
