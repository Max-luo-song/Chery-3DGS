#!/bin/bash

# Usage: ./retry.sh <command>

max=3
interval=5
cmd="$@"

n=0
until $cmd; do
  n=$((n+1))
  if [ $n -ge $max ]; then
    echo "Command failed after $max attempts."
    exit 1
  fi
  echo "Command failed. Retry #$n in $interval seconds..."
  sleep $interval
done
