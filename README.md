# bertopic_ecc
Master Thesis Project, analyze ECC with BERTopic

Steps to get ready:

1) Optional if using github: clone the repository

2) Open repository in your IDE of choice

3) Install Python >3.9 and create a virtual environment (optional):
conda create --name myenv

4) Install packages from requirements
pip install -r "(path)/.requirements.txt"
if you receive errors here, use the installation script which skips bad lines: package_installations.py 
by running: python package_installations.py

5) Change the following Variables in the config file:
folderpath_ecc = "D:/daten_masterarbeit/Transcripts_Masterarbeit_full/"   
index_file_ecc_folder = "D:/daten_masterarbeit/"
sample_size = 10 # number of unique companies to be analyzed, max is 1729

Note: If the Config file is not found, you need to set its filepath manually
in the file which you are attempting to run.

6) Run the bertopic_training.py script to train and save the BERTopic Model.

7) Run bertopic_fitting_visus.py to fit the model on data and generate visualizations + statistics. 

8) Run (TODO Skriptnamen hier noch einfügen) xxx to prepare the final dataset for the analysis. 

9) Run variable_analysis.py to perform statistical analysis and get results.

10) Optional: To create plots and statistics from the dataset, run run_ecc_analysis.py script. Some plots might be computational intensive, 
    so comment out/ comment in them in the main function before running the script. 

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


neccessary adjustments for zeroshot topic modeling:

min df = 0.01, min cluster size should also be low. 
also maybe umap n neighbors should be low (2)

intuition: größeres n neighbors: cluster sollten größer werden


Install for cuML: 
!pip install bertopic
!pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
!pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
!pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com
!pip install --upgrade cupy-cuda11x -f https://pip.cupy.dev/aarch64



    "sampling_mode": "full_random",  // Options: "full_random" or "random_company"