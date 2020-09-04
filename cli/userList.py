# userList.py - This file prints a csv table of all users with summary information
# Note: you will to export env variables MONGODB_URI & JWT_SECRET_KEY(not used, you can set to anything)
# usage: > py userList.py
# Sample output in ../samples/

import os,sys
sys.path.insert(1, os.path.realpath(os.path.pardir))
import config as cfg
import pandas as pd
from pymongo import MongoClient

dbClient = MongoClient(cfg.MONGODB_URI,retryWrites=False)  
userData = dbClient[dbClient.get_default_database().name].userdata 

pipeline = [ 
	{'$project': { '_id': 0,
                    'alias': 1, 
                    'demographic': 1,	
                    #'email': 1,
                    'train digits': { '$size':"$trainLog" },
                    'score digits': { '$cond': { 'if': { '$isArray': "$scoreLog" }, 'then': { '$size': "$scoreLog" }, 'else': 0} },
                    'resultReference':1,
                    'scoreThreshold': 1
                }}
]
user = userData.aggregate(pipeline)

df_userlist = pd.DataFrame(list(user))
df_userlist = pd.concat([df_userlist, pd.json_normalize(df_userlist.demographic)], axis = 1, sort=False) #convert demographic json to new columns
df_userlist.drop(['demographic'], axis=1, inplace=True)
print(df_userlist)
print("Number of participants: " + str(df_userlist.shape[0]))
print(df_userlist.to_csv(index=False))
