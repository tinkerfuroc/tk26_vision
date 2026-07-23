# 1 Hz health snapshot. Spec §4.4.
builtin_interfaces/Time last_pair_stamp
float32 color_age_sec      # now - last color header stamp (-1 if never seen)
float32 depth_age_sec      # now - last depth header stamp (-1 if never seen)
float32 pair_age_sec       # now - last synced pair stamp (-1 if never)
float32 sync_fps           # synced-pair rate over the last status window
uint64 pair_seq
