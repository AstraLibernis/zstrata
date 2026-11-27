#!/bin/bash
# profiles.sh: the same short zstrata rows under each GPU power setting (level, profile).
# Always puts the card back on auto / profile 0, also when interrupted.
G=$(for c in /sys/class/drm/card?; do [ "$(cat $c/device/device)" = 0x7550 ] && echo $c/device; done)
H=$(echo $G/hwmon/hwmon*)
restore() { echo auto > $G/power_dpm_force_performance_level; echo 0 > $G/pp_power_profile_mode; }
trap restore EXIT INT TERM
cd ~/workspace/zstrata && . tools/env.sh
out=~/workspace/notes/zstrata-run/1127
for setting in "$@"; do
  # level:profile[:tag], profile may be a whole CUSTOM line ("6 0 0 4 3300 ...")
  IFS=: read level prof tag0 <<< "$setting"
  echo $level > $G/power_dpm_force_performance_level || exit 1
  echo $prof > $G/pp_power_profile_mode || exit 1
  sleep 20
  tag=${tag0:-$level-p$prof}
  echo "== $tag idle: sclk $(( $(cat $H/freq1_input)/1000000 )) MHz, mclk $(( $(cat $H/freq2_input)/1000000 )) MHz, $(( $(cat $H/power1_average)/1000000 )) W, level $(cat $G/power_dpm_force_performance_level)"
  for part in "cross --only chain-long,chains,nets" "gpu --only chase,lds" "link --only roundtrip,wait,upload,download"; do
    zig-out/bin/zstrata $part $EXTRA 2>/dev/null
  done | grep -v "^device" > $out/prof-$tag$SUFFIX.csv
  echo "   rows $(wc -l < $out/prof-$tag$SUFFIX.csv)"
done
