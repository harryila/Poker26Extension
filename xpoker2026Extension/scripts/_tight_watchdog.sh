#!/usr/bin/env bash
WATCHDOG_LOG=/root/Poker26Extension/xpoker2026Extension/logs/overnight_wrapper_watchdog.log
echo "[tight_watchdog] started $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$$" >> "$WATCHDOG_LOG"
while true; do
    sleep 20
    usage_kb=$(df --output=used /dev/shm | tail -1)
    usage_gb=$(awk -v k="$usage_kb" 'BEGIN{printf "%.1f", k/1024/1024}')
    models=( /dev/shm/.hf_cache/hub/models--* )
    nmodels=0
    for d in "${models[@]}"; do
        [[ -d "$d" ]] && nmodels=$((nmodels+1))
    done
    if (( $(awk -v u="$usage_gb" 'BEGIN{print (u>20.0)?1:0}') )) && (( nmodels > 1 )); then
        oldest=""
        oldest_ts=9999999999
        now_s=$(date +%s)
        for d in "${models[@]}"; do
            [[ -d "$d" ]] || continue
            latest=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
            latest_int=${latest%.*}; latest_int=${latest_int:-0}
            age=$(( now_s - latest_int ))
            (( age < 30 )) && continue
            if (( latest_int < oldest_ts )); then
                oldest_ts=$latest_int
                oldest="$d"
            fi
        done
        if [[ -n "$oldest" ]]; then
            sz=$(du -sh "$oldest" 2>/dev/null | awk '{print $1}')
            echo "[tight_watchdog] $(date -u +%Y-%m-%dT%H:%M:%SZ) usage=${usage_gb}G nmodels=$nmodels purging idle: $(basename "$oldest") ($sz)" >> "$WATCHDOG_LOG"
            rm -rf "$oldest"
        fi
    fi
done
