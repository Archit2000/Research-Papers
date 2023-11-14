#!/bin/bash
echo "Enter Number of rows you want:"
read num
echo -e "Enter the sleep time you want:"
read sleeping
temp=1
while [ $temp -eq 1 ]
do
clear
echo ""
echo `date +%y/%m/%d_%H:%M:%S`
echo ""
ps -al | grep "python"
echo ""
op=`ps -al | grep -c "python"`
echo "Number of running python programs : $op"



echo -e "\n/Citeseer/0.05"
l=`wc -l '/home/archit/Thesis/Citeseer/0.05/parameter_readings.csv' | awk '{print $1}'`
echo "Number of lines : $l"
l_copy=`wc -l '/home/archit/Thesis/Citeseer/0.05/parameter_readings_copy.csv' | awk '{print $1}'`
diff=$(($l-$l_copy))
echo "Diff : $diff"
if [ $diff -gt 50 ]
then
echo "Greater than 50 diff"
echo "Make back up file"
cp "/home/archit/Thesis/Citeseer/0.05/parameter_readings.csv" "/home/archit/Thesis/Citeseer/0.05/parameter_readings_copy.csv"
fi
tail -n $num "/home/archit/Thesis/Citeseer/0.05/parameter_readings.csv"






echo -e "\n/Cora/0.05"
l=`wc -l '/home/archit/Thesis/Cora/0.05/parameter_readings.csv' | awk '{print $1}'`
echo "Number of lines : $l"
l_copy=`wc -l '/home/archit/Thesis/Cora/0.05/parameter_readings_copy.csv' | awk '{print $1}'`
diff=$(($l-$l_copy))
echo "Diff : $diff"
if [ $diff -gt 50 ]
then
echo "Greater than 50 diff"
echo "Make back up file"
cp "/home/archit/Thesis/Cora/0.05/parameter_readings.csv" "/home/archit/Thesis/Cora/0.05/parameter_readings_copy.csv"
fi
tail -n $num "/home/archit/Thesis/Cora/0.05/parameter_readings.csv"

sleep $sleeping
done
