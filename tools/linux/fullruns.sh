#!/bin/bash
# fullruns.sh TAG N [zstrata args]: N full zstrata runs under the custom GPU profile (GFXCLK min
# active 3300 MHz); the card goes back to auto / profile 0 at the end, also when interrupted.
G=$(for c in /sys/class/drm/card?; do [ "$(cat $c/device/device)" = 0x7550 ] && echo $c/device; done)
restore() { echo auto > $G/power_dpm_force_performance_level; echo 0 > $G/pp_power_profile_mode; }
trap restore EXIT INT TERM
tag=$1; n=$2; shift 2
out=~/workspace/notes/zstrata-run/1127
cd ~/workspace/zstrata && . tools/env.sh
[ "$PROFILE" = none ] || echo "${PROFILE:-6 0 0 4 3300 4 800 4587520 -65536 0}" > $G/pp_power_profile_mode || exit 1
sleep 5
for i in $(seq 1 $n); do
  scr=$(cat /sys/class/drm/card1-DP-3/enabled); lock=$(pgrep -c kscreenlocker_g)
  zig-out/bin/zstrata all "$@" > $out/$tag$i.csv 2> $out/$tag$i.err
  echo "$(date +%T) $tag$i exit=$? rows=$(wc -l < $out/$tag$i.csv) screen=$scr locker=$lock $(grep 'check,worst-move\|check,disturbed' $out/$tag$i.csv | cut -d, -f3,6 | tr '\n' ' ')"
done
