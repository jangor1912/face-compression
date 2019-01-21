#!/bin/bash
for a in {1..4}; do
    printf 'portrait.jpg 35.187.84.26 31169 5 '
done | xargs -n 4 -P 4 python pinger.py