## This shell script is used for inference 
## Input : sequence file path, result dir
## With --prediction sets True and --embeddings sets True, both inference results and embeddings of the model will be ouputted
conda activate covid_predict
python ./src/inference.py data/pVNT_seq.csv result --prediction True 
