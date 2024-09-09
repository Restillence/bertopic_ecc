# bertopic_ecc
Master Thesis Project, analyze ECC with BERTopic

Steps to get ready:

1) Optional if using github: clone the repository

2) Open repository in your IDE of choice

3) Install Python and create a virtual environment (optional):
conda create --name myenv

4) Install packages from requirements
pip install -r "(path)/.requirements.txt"

5) Change the following Variables in the config file:
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"   
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10 # number of unique companies to be analyzed, max is 1729

Note: If the Config file is not found, you need to set its filepath manually
in the file which you are attempting to run.


#useful videos to get started with bertopic:
https://www.youtube.com/watch?v=uZxQz87lb84
https://www.youtube.com/watch?v=5a5Dfft-rWc

###Requirements###
python >3.9
pandas 
numpy 
bertopic 

Additionally, for the Named Entity Recognition (NER) function, you'll need to download the Spacy English model:
python -m spacy download en_core_web_sm


###NOTE###
I used Visual Studio Code as IDE. Some part of the code might not work using another IDE 

###splitting methods have been checked manually on a sample###

###BERTopic functioning###
for every paragraph (default) or sentence a topic is assigned. 


Other things: config:
// Options: "regular", "iterative", "zeroshot", or "iterative_zeroshot"

    "embedding_model_choice": "finbert-pretrain",  # Choose between "all-MiniLM-L6-v2", "finbert-local", or "finbert-pretrain"
