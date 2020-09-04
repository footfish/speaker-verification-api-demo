# eer.py - This file generates a csv table of equal-error-rate benchmarks for all users in the database. 
# Note: you will to export env variables MONGODB_URI & JWT_SECRET_KEY(not used, you can set to anything)
# usage: > py eer.py
# TIP: set parameter VERBOSE = True to get more detail. 
# Sample output in ../samples/
import os,sys
sys.path.insert(1, os.path.realpath(os.path.pardir))
import config as cfg
import random
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle #persistance of gmm model
import pandas as pd

DISCARD_MFCC0=True #choose to discard first cepstral coefficient 
DIGIT_SET_SIZE = 3 #the number of digits in data set to use
SET_MATCH_DISCARD_THRESHOLD = 30 #A threshold used to determine if a digit set is 'valid' (really low match indicates recording error for legitimate speaker)
ENERGY_FILTER_THRESHOLD = 0.5 # (0-1 recommened) filters the MFCC features based on speakers  recorded energy level
VERBOSE = False #True gives more printout data

dbClient = MongoClient(cfg.MONGODB_URI,retryWrites=False)  
userData = dbClient[dbClient.get_default_database().name].userdata 

def main():

    #Get list of users for processing (excluding low scoreThreshold & training not completed) 
    userList = list(userData.find({'scoreThreshold': { '$gt' : -70}, 'trainProgress' : { '$gt' : 99} } ,{ 'alias':1}))
    print("Number of valid users:" + str(len(userList)))
    print ("Digit set size: " + str(DIGIT_SET_SIZE) )        
    print ("-----------------------------")            
    
    dfTotal = pd.DataFrame(columns = ["alias","FAR", "FRR", "eerRef", "userRef"])

    #for userIndex, user in enumerate(userList[0:1]):    #Uncomment to run for first only
    for userIndex, user in enumerate(userList):
        dfResult = pd.DataFrame(columns = ["type", "match"])
        
        #Create imposter list 
        imposterUserList = userList.copy()
        imposterUserList.pop(userIndex)

        #retrieve legitimate users trainLog & scoreLog
        UserDb = userData.find_one({'_id':user['_id']},{ "_id": 0, "alias": 1, "trainLog": 1, "scoreLog": 1, 'gmmPickleStore':1, 'scoreThreshold': 1, 'resultReference': 1})
        if UserDb == None or 'trainLog' not in UserDb: 
            print('error:this is unexpected, can not find user data')
        uDigitLog = extract_testLog(UserDb['trainLog'])
        if 'scoreLog' in UserDb: 
            uDigitLog += UserDb['scoreLog']
        print("User: '" + str(UserDb['alias']) + "' digits recorded: " + str(len(uDigitLog)))        
        setCounter = 1
        user_gmm=pickle.loads(UserDb['gmmPickleStore'])
        for log_entry in uDigitLog:
            if setCounter == 1: # 1st digit in set, initialise data 
                #Pick random imposter and get digits logged - iDigitLog (using trainLog only)
                ImposterDb = userData.find_one({'_id':random.choice(imposterUserList)['_id']},{ "_id": 0, "alias": 1, "trainLog": 1 })
                if ImposterDb == None or 'trainLog' not in ImposterDb: 
                    print('error:this is unexpected, can not find imposter data')
                iDigitLog = extract_testLog(ImposterDb['trainLog'])
                ilog_entry = iDigitLog[random.randint(0,len(iDigitLog)-1)] #choose random digit from imposter log 
                #initialise imposters set data 
                imfcc = np.array(ilog_entry['data']['mfcc'])
                iEnergy = np.array(ilog_entry['data']['energy'])
                iDigitSet = [ilog_entry['data']['digit']]
                #initialise users set data 
                umfcc = np.array(log_entry['data']['mfcc'])
                uEnergy = np.array(log_entry['data']['energy'])
                uDigitSet = [log_entry['data']['digit']]
                setCounter += 1
            elif setCounter == DIGIT_SET_SIZE: #collected digit set, now score 
                #user 
                np.append(umfcc,log_entry['data']['mfcc'],axis=0) 
                np.append(uEnergy,log_entry['data']['energy'],axis=0) 
                uDigitSet.append(log_entry['data']['digit'])
                if (DISCARD_MFCC0):
                    umfcc = np.delete(umfcc, 0, axis=1) #delete the 1st mfcc coeffecient 
                umfcc = np.delete(umfcc, np.nonzero(uEnergy < ENERGY_FILTER_THRESHOLD)[0],  axis=0) #apply energy mask 
                result = user_gmm.score_samples(umfcc)
                uScoreThreshold = UserDb['scoreThreshold']
                uMatchScore = 100*len(result[result>uScoreThreshold])/len(result)
                if uMatchScore > SET_MATCH_DISCARD_THRESHOLD: #Ignore badly recorded user sets 
                    dfResult = dfResult.append({'type': 'u','match': round(uMatchScore,2)}, ignore_index=True)
                    if VERBOSE: print("Usr: " + str(UserDb['alias']) + " Digits:" + str(uDigitSet) + " avg:" + str(round(np.average(result),2)) + " match:" + str(round(uMatchScore,2)) + "%")
                    #imposter 
                    ilog_entry = iDigitLog[random.randint(0,len(iDigitLog)-1)] #choose random digit from imposter log 
                    np.append(imfcc,ilog_entry['data']['mfcc'],axis=0) 
                    np.append(iEnergy,ilog_entry['data']['energy'],axis=0) 
                    iDigitSet.append(ilog_entry['data']['digit'])
                    if (DISCARD_MFCC0):
                        imfcc = np.delete(imfcc, 0, axis=1) #delete the 1st mfcc coeffecient 
                    imfcc = np.delete(imfcc, np.nonzero(iEnergy < ENERGY_FILTER_THRESHOLD)[0],  axis=0) #apply energy mask 
                    result = user_gmm.score_samples(imfcc) #score imposter against users gmm 
                    dfResult = dfResult.append({'type': 'i','match': round(100*len(result[result>uScoreThreshold])/len(result),2)}, ignore_index=True)
                    if VERBOSE: print("Imp: " + str(ImposterDb['alias']) + " Digits:" + str(iDigitSet) + " avg:" + str(round(np.average(result),2)) + " match:" + str(round(100*len(result[result>UserDb['scoreThreshold']])/len(result),2)) + "%")
                elif VERBOSE: print("*SKIPPED* Usr: " + str(UserDb['alias']) + " Digits:" + str(uDigitSet) + " avg:" + str(round(np.average(result),2)) + " match:" + str(round(uMatchScore,2)) + "%")
                #print("Energy frame size:" + str(len(uEnergy)) + " Over 1:" + str(len(uEnergy[uEnergy>1])))
                setCounter = 1
            else:
                setCounter += 1
                #user
                np.append(umfcc,log_entry['data']['mfcc'],axis=0) 
                np.append(uEnergy,log_entry['data']['energy'],axis=0) 
                uDigitSet.append(log_entry['data']['digit'])
                #imposter 
                ilog_entry = iDigitLog[random.randint(0,len(iDigitLog)-1)] #choose random digit from imposter log 
                np.append(imfcc,ilog_entry['data']['mfcc'],axis=0) 
                np.append(iEnergy,ilog_entry['data']['energy'],axis=0) 
                iDigitSet.append(ilog_entry['data']['digit'])

        #Calculate the eer 
        falseAccept=0 #imposter is allowed
        falseReject=1 #legitimate speaker is denied
        matchThreshold=100 

        while falseAccept < falseReject: 
            falseAccept=len(dfResult[(dfResult.match >= matchThreshold) & (dfResult.type == 'i')])
            falseReject=len(dfResult[(dfResult.match <= matchThreshold) & (dfResult.type == 'u')])
            matchThreshold -= 0.1

        dfTotal = dfTotal.append({'alias': UserDb['alias'],'FAR': round(100*falseAccept/len(dfResult),2), 'FRR': round(100*falseReject/len(dfResult),2), 'eerRef' : round(matchThreshold,2), 'userRef': UserDb['resultReference']}, ignore_index=True)

        print("False Accept Rate (FAR):" + str(round(100*falseAccept/len(dfResult),2)) + "% ", end = '') 
        print("False Reject Rate (FRR):" + str(round(100*falseReject/len(dfResult),2)) + "% " , end = '')
        print("Match Threshold:" + str(round(matchThreshold,2)) + "%")
        print ("-----------------------------")        
        #print(dfResult)

    print(dfTotal.to_csv(index=False))

def extract_testLog(trainLog):
    #The trainLog contains a store of all training 'feature vectors' along with a 'label' of the digit spoken. 
    #The trainLog store is split 50/50 for 'test data'/'train data'. 
    digitCounter = [0,0,0,0,0,0,0,0,0,0] #counter for each occurrance of digits 0-9 in trainLog. 
    testLog = []
    for log_entry in trainLog:
        if digitCounter[log_entry['data']['digit']] % 2 != 0:
            testLog.append(log_entry)
        digitCounter[log_entry['data']['digit']] += 1 
    return(testLog)


if __name__ == '__main__':
    main()
