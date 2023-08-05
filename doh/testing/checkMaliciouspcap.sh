rm -rf input
rm -rf output
mkdir input
mkdir output
echo Choose model for Level 1 Classification - DoH and Non-Doh
echo 1 RF 
echo 2 Decision Tree
echo 3 DNN
echo Choose betweeen 1-3 :
read m1
echo Choose model for Level 2 Characterization - Benign and Malicious
echo 1 RF
echo 2 Decision Tree
echo 3 DNN
echo Choose betweeen 1-3 :
read m2

cd ../meter
counter=1
for file in ../testing/pcaps/*
do
	PYTHONPATH=../ python3 dohlyzer.py -f $file -c ../testing/input/input_$counter.csv
	counter=`expr $counter + 1`
done
cd ../testing
python3 merge.py
python3 run.py $m1 $m2

