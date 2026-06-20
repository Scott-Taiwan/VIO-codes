TS=$(date +%Y%m%d_%H%M%S) && ./drone_localize --port /dev/ttyTHS1 --baud 57600 --zoom 19 --interval 3 2>&1 | tee ${TS}_stdout.txt &
