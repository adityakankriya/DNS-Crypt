cd ../meter
counter=1
for file in $(find ../pcaps -name "*.pcap")
do
	PYTHONPATH=../ python3 dohlyzer.py -f $file -c ../raw_data/csv/output_$counter.csv
	counter=`expr $counter + 1`
done
