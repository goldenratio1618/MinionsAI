#!/bin/bash
mkfifo fifo0 fifo1
python3 engine.py > fifo0 < fifo1 &
python3 randomAI.py < fifo0 > fifo1
rm fifo0 fifo1
