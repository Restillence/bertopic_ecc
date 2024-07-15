# bertopic_ecc
Master Thesis Project, analyze ECC with BERTopic

Steps to get ready:

1) Optional if using github: clone the repository

2) Open repository in your IDE of choice

3) Install Python and create a virtual environment (optional):
conda create --name myenv

4) Install packages from requirements
pip install -r "(path)/.requirements.txt"

5) Change the following Variables in the main script:
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"   
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10 # number of unique companies to be analyzed, max is 1729



#useful videos to get started with bertopic:
https://www.youtube.com/watch?v=uZxQz87lb84
https://www.youtube.com/watch?v=5a5Dfft-rWc

###Requirements###
python >3.9
pandas 
numpy 
bertopic 