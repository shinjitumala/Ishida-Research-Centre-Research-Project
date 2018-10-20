#!/bin/bash

calc=0
sum=0
count=0
while read line
do
  if [ "$line" = "===============================================" ]; then
    if [ $count -ne 0 ]; then
      avg=`echo "scale=6; $sum / $count " | bc`
      echo average_score: $avg
      sqavg=`echo "scale=6; $total / $count " | bc`
      div=`echo "scale=6; $sqavg - ( $avg * $avg ) " | bc`
      echo variance: $div
    fi
    echo $line
    calc=0
  elif  [[ $line =~ = ]]; then
    echo $line
    calc=1
    sum=0
    count=0
    total=0
  else
    val=`echo $line | cut -f 2 -d " "`
    count=$((count + 1))
    sum=`echo "scale=8; $sum + $val " | bc`
    total=`echo "scale=8; $total + ( $val * $val ) " | bc`
  fi


done < $1
