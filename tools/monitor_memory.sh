#!/bin/bash
# Memory monitoring script for macOS
# Run in a separate terminal while experiments are running
#
# Usage:
#   ./scripts/monitor_memory.sh [interval_seconds]

INTERVAL=${1:-5}

echo "Memory Monitor (updating every ${INTERVAL}s)"
echo "Press Ctrl+C to stop"
echo ""
echo "Timestamp           | Memory Used | Memory Free | Swap Used | Python Procs"
echo "-------------------+-------------+-------------+-----------+-------------"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get memory info (macOS)
    MEM_INFO=$(vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f Mi\n", "$1:", $2 * $size / 1048576);')
    
    # Extract key metrics
    MEM_USED=$(echo "$MEM_INFO" | grep -E "active|wired" | awk '{sum+=$2} END {printf "%.1f GB", sum/1024}')
    MEM_FREE=$(echo "$MEM_INFO" | grep "free" | awk '{printf "%.1f GB", $2/1024}')
    
    # Get swap usage
    SWAP_INFO=$(sysctl vm.swapusage | awk '{print $7}')
    
    # Count Python processes
    PYTHON_PROCS=$(ps aux | grep -E "python.*verify_correlation|python.*negation" | grep -v grep | wc -l | tr -d ' ')
    
    printf "%s | %11s | %11s | %9s | %11s\n" \
        "$TIMESTAMP" "$MEM_USED" "$MEM_FREE" "$SWAP_INFO" "$PYTHON_PROCS"
    
    sleep $INTERVAL
done
