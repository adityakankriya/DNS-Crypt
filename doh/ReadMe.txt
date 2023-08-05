This Project is done by Team HoneyHex
The members are 
1) Maj Navneet Sharma - 22111087
2) Maj Ashish Ahluwalia - 21111073
3) Aditya Kankriya - 22111072
4) Mitrajsinh Chavda - 22111077
5) Vamsee Krishna  - 22111065
6) Rumit Gore - 22111409

This project contains the following directories
1) code - Place where our codes for the project are kept
2) meter - A module to extract features from PCAPS
3) output - Where our output of our project is stored
4) pcaps - Where we have store the PCAPS captured for training
5) processed_data - Where we kept our processed file which can be use for training
6) raw_data - A place where we keep our generated csv in raw form
7) systems - A place where we have stored the trained models
8) testing - A place where you can check whether pcaps files are malicious or not
9) zips - A place where we keep our zips which contains pcap files

To understand the whole project sequence, we will go in chronological order 

step 1: PCAP FILES:
Various pcasp files containing doh tunnel and non doh tunnel data are stored in pcap folder, these pcap files were extracted from adgaurd,
cloudflare,quad9 and googleDNS and are stored in one folder.

Step 2: GENERATING FEATURES
Now next important thing is to generate meaningful data from pcap files. To do so, we should use pcaps files generated from last step.
So to extract feature we need to go to raw_data folder. There we will find a script file which will take all pcaps files and extract features and stored in csv folder in raw_data
To run it
>> cd raw_data
>> sh generate-feature.sh

Step 3: Generating Process Data
From the generated csv from pcaps in Step 2, now we need to merge all the data and label it according to our need. So we have processed them and store the files in processed_data folder.

Step 4: Training Our Model
After generating the required data, next step is to run the training algorithms for the classifiers for L1 and L2 level. For that we have two directory in our code folder. 
To train level 1, we have to use the following codes
>> cd code
>> cd 1_classification
>> sh run-classification.sh

To train level 2, we have to use the following codes
>> cd code
>> cd 2_characterization
>> sh run-charavterization.sh

And after training, we have store those models in systems directories. So that we can use them later.
Also, we have stored the output of the models in output directory.

Step 5: Testing on Pcaps
So now our final step. Given any pcap files, we have identify whether it is malicious or not. 
To do this, we have created testing folder. 
Inside it, we have a folder named pcaps. Where we have to place all pcaps files that we need to test.
After placing them, we need to run the script to test it.
To test the pcaps we need
>> cd testing
>> sh checkMaliciouspcap.sh

And this will test the pcaps, and store the output of our testing in output directory.

Also we have provided the requirements file which helps to fill the dependencies
To install them 
>> pip install -r requirements.txt
 for more assistance in running we have also provided a video - doh.webm
