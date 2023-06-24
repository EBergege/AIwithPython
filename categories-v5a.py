from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.llm import LangChainInterface

from dotenv import load_dotenv

import os
import time
import numpy as np
import pandas as pd
import json
import logging

GLOBALFILEDIR = './'
INIT_PROG_FILE = GLOBALFILEDIR+'categorize.json'
CATEGORY_SEPARATOR = '-#-'

def read_promptTemplate():
    with open( PROMPT_TEMPLATE_FILE ) as f:
        lines = f.readlines()
    return ''.join(lines)

def save_CategorizedComments(result_df):
    result_df.to_csv(RESULT_FILE,index=False)

def buildGraph(graphLabel,a_dataframe):
    graph_filepath = RESULT_GRAPH_FILE
    p_table = pd.pivot_table( a_dataframe , index=['categories'], values=SOURCE_FILE_COMMENT_COLUMN, aggfunc=lambda x: len(x.unique()))
   
    p_table = p_table.sort_values('categories',ascending=False)  # to sort categories bars in graph
    
    import matplotlib.pyplot as plt
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    ax = p_table.plot(kind='barh', figsize=(1400*px, 800*px), color='#86bf91',  width=0.85)
    
    ax.set_xlabel('Number of comments')
    ax.figure.legend().remove()
            
    for bar, nbcomments in zip(ax.patches, p_table[SOURCE_FILE_COMMENT_COLUMN]):
        ax.text(bar.get_width()/2, bar.get_y()+bar.get_height()/2, nbcomments, color = 'black', ha = 'left', va = 'center', weight = 'bold')
    
    ax.figure.tight_layout()
    ax.figure.savefig(graph_filepath)
    ax.figure.show()       

def read_realNPSsurveyFile():
    if os.path.exists(SOURCE_FILE):    
        tmp_df = pd.read_csv(SOURCE_FILE, engine='python', on_bad_lines='warn')
        tmp_df = tmp_df.drop(axis=0, index=0) # delete first csv record , with empty fields
        tmp_df = tmp_df.iloc[:, [1,2,7,9,10]]
        # 1 "Survey Invitation",
        # 2 "Invitation Link", 
        # 7 "How likely are you to reco mmend CODE to a colleague? 
 #0 (Not at all likely) ------------------------------------------------> 10 (Extremely likely)",
        # 9 "Pleas e share your sugg estions for improving CODE."
        #10 "Please select your role.""
        
        tmp_df.insert(loc=0, column='user_id', value = (tmp_df['Survey Invitation'].str[-18:])) 
        tmp_df.insert(loc=1, column='categories' , value=' ')
        tmp_df.insert(loc=tmp_df.columns.get_loc('categories'),
                      column='comment_id',
                      value = (tmp_df['Invitation Link'].str[-36:]))  # take initial categories column position 
        tmp_df = tmp_df.iloc[:, [0,1,2,5,6,7]]
        tmp_df = tmp_df.reset_index()  # make sure indexes pair with number of rows, after row deletion
        
        global SOURCE_FILE_COMMENT_COLUMN
        SOURCE_FILE_COMMENT_COLUMN = tmp_df.columns[5]  # should contain "Pleas e share your sugg estions for improving CODE."
        
        return tmp_df
    else:
        return 

def isMultiCategories(acomment):
    sep_pos=acomment.find(CATEGORY_SEPARATOR, 0)
    if sep_pos>0:
        return True
    else:
        return False
 
def getParameters():
    problemFound = False
    try:   
        with open( INIT_PROG_FILE , mode="r") as dataFile:
            userFiledata=json.load(dataFile)  
    except FileNotFoundError:
        print(f"No parameter file {INIT_PROG_FILE}. End of run.")
        problemFound = True
        #TODO: create a default new categorize.json file !!!
    else:
        param_key = "source_file"
        if param_key in userFiledata:
            global SOURCE_FILE
            SOURCE_FILE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} .")
            problemFound = True
        
        param_key = "prompt_template_file"
        if param_key in userFiledata:
            global PROMPT_TEMPLATE_FILE
            PROMPT_TEMPLATE_FILE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True
            
        param_key = "result_file"
        if param_key in userFiledata:
            global RESULT_FILE
            RESULT_FILE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True
        
        param_key = "result_graph_file"
        if param_key in userFiledata:
            global RESULT_GRAPH_FILE
            RESULT_GRAPH_FILE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True
        
        param_key = "user_info_file"
        if param_key in userFiledata:
            global GLOBAL_EXCEL_USER_INFO_FILE
            GLOBAL_EXCEL_USER_INFO_FILE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True  
            
        param_key = "model"
        if param_key in userFiledata:
            global GLOBAL_PARAMS_MODEL
            GLOBAL_PARAMS_MODEL = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True  
        
        param_key = "decode_method"
        if param_key in userFiledata:
            global GLOBAL_PARAMS_DECODE
            GLOBAL_PARAMS_DECODE = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True  
        
        param_key = "param_rep_penalty"
        if param_key in userFiledata:
            global GLOBAL_PARAMS_PENALTY
            GLOBAL_PARAMS_PENALTY = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True
        
        param_key = "param_min_token"
        if param_key in userFiledata:
            global GLOBAL_PARAMS_MIN_TOKEN
            GLOBAL_PARAMS_MIN_TOKEN = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True  
         
        param_key = "param_max_token"
        if param_key in userFiledata:
            global GLOBAL_PARAMS_MAX_TOKEN
            GLOBAL_PARAMS_MAX_TOKEN = userFiledata[param_key]
        else:
            print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
            problemFound = True 
            
        if GLOBAL_PARAMS_DECODE != "greedy":
            param_key = "param_temperature"
            if param_key in userFiledata:
                global GLOBAL_PARAMS_TEMP
                GLOBAL_PARAMS_TEMP = userFiledata[param_key]
            else:
                print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
                problemFound = True  
            
            param_key = "param_top_p"
            if param_key in userFiledata:
                global GLOBAL_PARAMS_TOPP
                GLOBAL_PARAMS_TOPP = userFiledata[param_key]
            else:
                print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
                problemFound = True 
                
            param_key = "param_top_k"
            if param_key in userFiledata:
                global GLOBAL_PARAMS_TOPK
                GLOBAL_PARAMS_TOPK = userFiledata[param_key]
            else:
                print(f"Missing entry '{param_key}' in parameter file {INIT_PROG_FILE} . ")
                problemFound = True 
                
    if problemFound == True:
        print("End of run")
        quit()
    else:    
        print(f'Parameters loaded from init file: {INIT_PROG_FILE}')     

def storeRunningNumber( runningNumber ):    
    param_key = "last_job_running_number"
    param_data_line={ param_key : runningNumber }
    try:       
        with open( INIT_PROG_FILE , mode="r") as dataFile:
            existingdata=json.load(dataFile)     
    except FileNotFoundError:
        with open( INIT_PROG_FILE , mode="w") as dataFile:
            json.dump(param_data_line, dataFile, indent=4)
    else:
        existingdata.update(param_data_line)
        with open( INIT_PROG_FILE , mode="w") as dataFile:
            json.dump(existingdata, dataFile, indent=4) 

def getLastRunNumber():
    param_key = "last_job_running_number"
    default_run_number = 0
    param_data_line={ param_key : default_run_number }
    try:   
        with open( INIT_PROG_FILE , mode="r") as dataFile:
            userFiledata=json.load(dataFile) 
    except FileNotFoundError:
        print(f"No parameter file {INIT_PROG_FILE} , creating it...")
        with open( INIT_PROG_FILE , mode="w") as dataFile:
            json.dump(param_data_line, dataFile, indent=4)    
    else:
        if param_key in userFiledata:
            last_running_number = userFiledata[param_key]
            return last_running_number     
        else:
            # in case number field is empty, set it to 0
            storeRunningNumber( 0 )
            return 0
    return ''
    
def getCurrentDay():
    return time.strftime('%Y%m%d', time.gmtime())
    
def setRunningKey():
    last_running_number = getLastRunNumber()
    new_running_number = int(last_running_number) + 1 
    
    storeRunningNumber(new_running_number)
    newRunningKey = getCurrentDay() + str(new_running_number)
    
    print(f"Running key: {newRunningKey}")
    return newRunningKey

def computeNewResultFileName( aCSVFileName ) :
    extension = aCSVFileName[-4:]  # get last right most characters
    if extension[0] == ".":
        existingfile = aCSVFileName[:-4]  # remove ending .csv from source filename
        return existingfile+GLOBAL_RUNNING_KEY+extension
    else:
        return existingfile+GLOBAL_RUNNING_KEY   

def read_user_info_file():
    # require_cols = [0, 3]
    # only read specific columns from an excel file
    required_df = pd.read_excel( GLOBAL_EXCEL_USER_INFO_FILE
                                ,sheet_name = "CODE User List"
                                ,usecols = "C,V,W,AB"
                                #,header=
                                #,skiprows=1
                                #,names = ["USER_ID","GEO","MARKET","BU"]
                                ) 
    print("end of reading excel file")
    # print(required_df)
    return required_df

start = time.time()
print("Starting Categorization process")
print(time.strftime('%H:%M:%S', time.gmtime(start)))
             
GLOBAL_RUNNING_KEY = setRunningKey()

getParameters()

print("Running job with these files:")
print(f"..prompt template file: {PROMPT_TEMPLATE_FILE}")
print(f"..source file         : {SOURCE_FILE}")
RESULT_FILE = computeNewResultFileName(RESULT_FILE)
print(f"..resulting csv file  : {RESULT_FILE }")
RESULT_GRAPH_FILE = computeNewResultFileName(RESULT_GRAPH_FILE)
print(f"..resulting graph file: {RESULT_GRAPH_FILE}")
print(f"..Excel User file       : {GLOBAL_EXCEL_USER_INFO_FILE}")

# running info in a log file
logging.basicConfig(filename=f"./log_{GLOBAL_RUNNING_KEY}.log", level=logging.INFO)
logging.info(f"Execution start: {time.strftime('%H:%M:%S', time.gmtime(start))}")

load_dotenv()

NbCommentNotCategorized = 0

if GLOBAL_PARAMS_DECODE == "greedy":
    params = GenerateParams(
        decoding_method = GLOBAL_PARAMS_DECODE,
        repetition_penalty = GLOBAL_PARAMS_PENALTY,
        min_new_tokens = GLOBAL_PARAMS_MIN_TOKEN,
        max_new_tokens = GLOBAL_PARAMS_MAX_TOKEN
    )
    logging.info(f"Params: model: {GLOBAL_PARAMS_MODEL} - decoding:{GLOBAL_PARAMS_DECODE} ")
    logging.info(f"   - rep_penalty:{GLOBAL_PARAMS_PENALTY} - min_tok={GLOBAL_PARAMS_MIN_TOKEN} - max_tok={GLOBAL_PARAMS_MAX_TOKEN}")
else:
    params = GenerateParams(
        decoding_method = GLOBAL_PARAMS_DECODE,
        temperature = GLOBAL_PARAMS_TEMP, 
        top_p = GLOBAL_PARAMS_TOPP, 
        top_k = GLOBAL_PARAMS_TOPK,
        repetition_penalty = GLOBAL_PARAMS_PENALTY,
        min_new_tokens = GLOBAL_PARAMS_MIN_TOKEN,
        max_new_tokens = GLOBAL_PARAMS_MAX_TOKEN
    )
    logging.info(f"Params: model: {GLOBAL_PARAMS_MODEL} - decoding:{GLOBAL_PARAMS_DECODE}  ")
    logging.info(f"   - temperature:{GLOBAL_PARAMS_TEMP} - top_p:{GLOBAL_PARAMS_TOPP} - top_k:{GLOBAL_PARAMS_TOPK} - rep_penalty:{GLOBAL_PARAMS_PENALTY} ")
    logging.info(f"   - min_tok={GLOBAL_PARAMS_MIN_TOKEN} - max_tok={GLOBAL_PARAMS_MAX_TOKEN}")

logging.info(f"prompt template file: {PROMPT_TEMPLATE_FILE} ")
logging.info(f"result file: {RESULT_FILE} ")
logging.info(f"image file: {RESULT_GRAPH_FILE} ")

credentials = Credentials(os.environ['GENAI_KEY'], api_endpoint=os.environ['GENAI_API'])
llm = LangChainInterface(credentials=credentials, model=GLOBAL_PARAMS_MODEL, params=params)

from langchain import PromptTemplate

template = read_promptTemplate()
prompt = PromptTemplate(template=template, input_variables=["user_feedback"])

read_comments_df = read_realNPSsurveyFile()

# because we may have duplicated comment for each potential categories, 
# we need to work in another dataframe to build row by row
updated_comments_df = pd.DataFrame(columns=['user_id','comment_id','categories', SOURCE_FILE_COMMENT_COLUMN, 'Please select your role.'])

from langchain import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

index_result_file = 0
for index, row in read_comments_df.iterrows():

    user_id=row['user_id']
    comment_id=row['comment_id']
    user_feedback = row[SOURCE_FILE_COMMENT_COLUMN]
    user_role=row['Please select your role.']
    
    if isinstance(user_feedback,str)==False:   # in case of weird empty/nan feedback ?
        pass
    elif len(user_feedback)>0 and user_feedback != ' ':
        return_Categories = llm_chain.run(user_feedback)
      
        if isMultiCategories(return_Categories):
            one_go = [a_category.strip() for a_category in return_Categories.split( CATEGORY_SEPARATOR )]
        
            for a_cat in one_go: 
                if a_cat=='':     
                    #print(f"a_cat vide pour index:{index} , return_categories:{return_Categories}") 
                    pass   
                else:                  
                    new_row = {'user_id': user_id,
                               'comment_id': comment_id,
                               'categories': a_cat,
                               SOURCE_FILE_COMMENT_COLUMN: user_feedback,
                               'Please select your role.' : user_role
                            }      
                
                    updated_comments_df.loc[index_result_file] = new_row   # Use the loc method to add the new row to the DataFrame
                    index_result_file = len(updated_comments_df)
                
                    #print(f"MULT CATEGORY FOUND - index:{updated_comments_df['comment_id'][index_result_file-1]} categories:{updated_comments_df['categories'][index_result_file-1]} feedback: {updated_comments_df[SOURCE_FILE_COMMENT_COLUMN][index_result_file-1]}")        
                    print(f"Index:{index} - Comment id:{updated_comments_df['comment_id'][index_result_file-1]} (M)")
        else:
            new_row = {'user_id': user_id,
                       'comment_id': comment_id,
                       'categories': return_Categories,
                       SOURCE_FILE_COMMENT_COLUMN: user_feedback,
                       'Please select your role.' : user_role
                       }
        
            updated_comments_df.loc[len(updated_comments_df)] = new_row   # Use the loc method to add the new row to the DataFrame
            index_result_file = len(updated_comments_df) 
            #print(f"{comment_id}-{user_feedback[0:100]}  ... result--> {llm_chain.run(user_feedback)}")
            #print(f"index:{updated_comments_df['comment_id'][index]} categories:{updated_comments_df['categories'][index]} feedback: {updated_comments_df[SOURCE_FILE_COMMENT_COLUMN][index]}")
            print(f"Index:{index} - Comment id:{updated_comments_df['comment_id'][index_result_file-1]}")
        if return_Categories == "Not classified":
            NbCommentNotCategorized=NbCommentNotCategorized+1
            
# if os.path.exists( GLOBAL_EXCEL_USER_INFO_FILE ):
#     # join with user records
#     tmp_xl = read_user_info_file()
#     finalComments_df = pd.merge( updated_comments_df, tmp_xl, how='left',left_on='user_id',right_on='USER_ID')
#     finalComments_df = finalComments_df.drop(columns=["USER_ID"]) # not needed column
# else:
#     finalComments_df = updated_comments_df
finalComments_df = updated_comments_df

print('Saving categories in RESULT comment file.')   
save_CategorizedComments(finalComments_df)
logging.info(f"Not classified comments: {NbCommentNotCategorized}")

#TODO: build graph for each dimensions
buildGraph(graphLabel=SOURCE_FILE , a_dataframe=finalComments_df)

end = time.time()
print(f"End of Categorization process: {time.strftime('%H:%M:%S', time.gmtime(end))}.")
logging.info(f"Success execution - finished: {time.strftime('%H:%M:%S', time.gmtime(end))}.")      
print(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(end-start))}.")