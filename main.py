"""
This file contains the main function of the program.
I use BERTtopic to analyze ECC data. 

"""
#imports
import pandas as pd

#variables
folderpath_ecc = "daten_masterarbeit/Transcripts_Masterarbeit/"   
index_file_ecc_folder = "daten_masterarbeit/"
#samplesize = 1000 # number of documents to be analyzed


#constants
#nothing to change here
index_file_path = index_file_ecc_folder+"list_earnings_call_transcripts.csv"

#read index file to pandas dataframe
index_file = pd.read_csv(index_file_path)
print(index_file.head(5))


def locate_file():
    pass


def build_data_pipeline_ecc():   #this function might go to a submodule
    pass

def match_ecc_financial_data(): #this function might go to a submodule
    pass

def main():
    pass


if __name__ == "__main__":
    main()