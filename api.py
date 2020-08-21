import config as cfg
import numpy as np
from flask import Flask, abort, render_template, request
from flask_restful import reqparse,Resource, Api
from flask_jwt_extended import create_access_token,JWTManager,jwt_required, get_jwt_identity, get_raw_jwt, get_jti, decode_token
import datetime
from sklearn import mixture
import pickle #persistance of gmm model
from os import path
from os import remove
from time import time
from pymongo import MongoClient
from bson import Binary

dbClient = MongoClient(cfg.MONGODB_URI,retryWrites=False)  
userData = dbClient[dbClient.get_default_database().name].userdata 

app = Flask(__name__)
api = Api(app)
app.config['JWT_SECRET_KEY'] = cfg.JWT_SECRET_KEY
jwt = JWTManager(app) #initialise JWT 

N_GMMCOMPONENTS=5
DISCARD_MFCC0=True #choose to discard first cepstral coefficient 
TRAINLOGSIZE=60 #the maximum number of digits held in the training log (circular log)
SCORELOGSIZE=60 #the maximum number of digits held in the scoring log (circular log)
TRAINEDSIZE=60 #the number of digits in the training log considered 100% trained 

parser = reqparse.RequestParser()
parser.add_argument('mfcc', type=list, location='json')
parser.add_argument('energy', type=list, location='json')
parser.add_argument('username')
parser.add_argument('alias')
parser.add_argument('demographic', type=dict, location='json')
parser.add_argument('clientInfo')

# Provide api information
class ApiInfo(Resource):
    def get(self):
        return {'version': '0.1',
                'usage': '/doc /user /login /status /train /benchmark /score /delete'
                }

# ' user' and return JWT
class Login(Resource):
  def post(self):
    args = parser.parse_args()
    user = userData.find_one({"email": args['username']})
    if user != None:
        print(args['username'] + ' logged in')
        expires = datetime.timedelta(days=7)
        access_token = create_access_token(identity=user['userId'], expires_delta=expires)
        session_log_item = { 'timestamp': time(), 'session': decode_token(access_token)['jti'], 'clientInfo': args['clientInfo'] }
        userData.update_one({'userId':user['userId']},{'$push': {'sessionLog':  session_log_item}})
        return {'token': access_token}, 200
    else:
        return {'error': 'Email invalid'}, 401

# 'Register user' and return JWT
class Register(Resource):
  def post(self):
    args = parser.parse_args()
    #validate 
    if args['username'] == "":
        return {'error': 'Email invalid'}, 401


    print(args['username'] + ' registering')
    user = userData.find_one({"email": args['username']})

    if user == None:
        userData.insert_one({ "email": args['username'], "userId":hash(args['username']), "alias": args['alias'], "demographic": args['demographic'] })
        print(args['username'] + ' registered')
        expires = datetime.timedelta(days=7)
        access_token = create_access_token(identity=hash(args['username']), expires_delta=expires)
        return {'token': access_token}, 200
    else:
        return {'error': 'Already exists'}, 403


# Delete: Deletes trainLog store & GMM pickle & resets stats. 
class Delete(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        userData.update_one({'userId':JWT_userId},{'$unset': {'gmmPickleStore': ""}})
        userData.update_one({'userId':JWT_userId},{'$unset': {'trainLog': ""}})
        userData.update_one({'userId':JWT_userId},{'$unset': {'scoreLog': ""}})
        userData.update_one({'userId':JWT_userId},{'$set': {'trainDataLength': 0, 'testDataLength': 0, 'trainProgress': 0}})
        reply = {
        'deleted' : True,    
        }
        return reply, 200

# Status: Returns training status
class Status(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, 'alias':1, 'trainDataLength':1,'testDataLength':1, 'trainProgress':1 , 'trainLog':1, 'scoreLog':1, 'resultReference': 1,'scoreThreshold': 1 })
        if db_result != None: 
            if 'trainDataLength' in db_result:
                train_len=db_result['trainDataLength']
            else:
                train_len = 0
            if 'testDataLength' in db_result:
                test_len=db_result['testDataLength']
            else: 
                test_len = 0
            if 'trainProgress' in db_result:
                train_prog=db_result['trainProgress']
            else:
                train_prog = 0
            if 'scoreThreshold' in  db_result:
                scoreThreshold=db_result['scoreThreshold'] 
            else: 
                scoreThreshold=0
            if 'resultReference' in  db_result:
                resultReference=db_result['resultReference'] 
            else: 
                resultReference=0
            train_log_summary = []
            if 'trainLog' in db_result:
                #build summary trainlog returned in api 
                train_log_summary_item = {}
                for log_entry in db_result['trainLog']:
                    train_log_summary_item['datetime'] = str(datetime.datetime.fromtimestamp(log_entry['timestamp']).isoformat())
                    train_log_summary_item['digit']= log_entry['data']['digit']
                    train_log_summary.append(train_log_summary_item.copy())
            score_log_summary = []
            if 'scoreLog' in db_result:
                #build summary scorelog returned in api 
                score_log_summary_item = {}
                for log_entry in db_result['scoreLog']:
                    score_log_summary_item['datetime'] = str(datetime.datetime.fromtimestamp(log_entry['timestamp']).isoformat())
                    score_log_summary_item['digit']= log_entry['data']['digit']
                    score_log_summary_item['result']= log_entry['result']                    
                    score_log_summary.append(score_log_summary_item.copy())
            reply = {
            'alias': db_result['alias'],
            'train_data_length' : train_len,
            'test_data_length' : test_len,
            'training_progress' : train_prog,
            'training_log' : train_log_summary[::-1],      
            'scoring_log' : score_log_summary[::-1],                  
            'score_threshold': scoreThreshold, 
            'result_reference' : resultReference
            }
            return reply, 200
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500


# Training - train model 
class Train(Resource):
    @jwt_required    
    def put(self):
        JWT_userId = get_jwt_identity()
        JWT_token = get_raw_jwt()
        #read mfcc features from resource
        args = parser.parse_args()
        #mfcc = np.array(args['mfcc'])     
        #basic sanity check 
        if len(args['mfcc']) < 1 or len(args['energy']) < 1 or len(args['mfcc']) != len(args['energy']): 
            return {'error': 'data malformed - check the API specification'}, 400

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 1, "trainLog": 1})
        if db_result == None: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        #get trainLog and add new data 
        if 'trainLog' in db_result:
            train_log = db_result['trainLog']
        else:
            train_log=[]        
        train_log_item = { 'timestamp': time(), 'session':JWT_token['jti'], 'data': request.json }
        train_log.append(train_log_item.copy())
        #trim if log too big 
        if len(train_log) > TRAINLOGSIZE:
            train_log.pop(0)
            
        feature_data =  extract_features(train_log) 

        #calculate training progress 
        api_training_progress = int(100*len(train_log)/TRAINEDSIZE)  #fully traind when TRAINEDSIZE digits recorded. 

        #build summary trainlog returned in api 
        train_log_summary = []
        train_log_summary_item = {}
        for log_entry in train_log:
            train_log_summary_item['datetime'] = str(datetime.datetime.fromtimestamp(log_entry['timestamp']).isoformat())
            train_log_summary_item['digit']= log_entry['data']['digit']
            train_log_summary.append(train_log_summary_item.copy())

        # create model
        gmm = mixture.GaussianMixture(N_GMMCOMPONENTS, covariance_type='full')

        # TODO/check 
        #prune mfcc file if too large         
        #fit (warm start may help)

        tTime=time()
        gmm.fit(feature_data['train_data'])
        tTime=time()-tTime

        #score 
        if len(feature_data['test_data']):
            testResult = gmm.score_samples(feature_data['test_data'])
            scoreThreshold=np.average(testResult)-np.std(testResult)
            resultReference=round(100*len(testResult[testResult>scoreThreshold])/len(testResult)) # % of matched frames in test data set
        else:
            scoreThreshold=0
            resultReference=0
            
        #Update db - gmm pickle / cache stats / train_log
        userData.update_one({'userId':JWT_userId},{'$set': {'gmmPickleStore': Binary(pickle.dumps(gmm)), 'resultReference': resultReference,'scoreThreshold': scoreThreshold,'trainDataLength': len(feature_data['train_data']), 'testDataLength': len(feature_data['test_data']), 'trainProgress': api_training_progress, 'trainLog': train_log}})

        reply = {
        'training_progress' : api_training_progress,
        'train_data_length' : len(feature_data['train_data']),
        'test_data_length' : len(feature_data['test_data']),
        'training_time' : tTime,
        'training_log' : train_log_summary[::-1],
        'score_threshold': scoreThreshold, 
        'result_reference' : resultReference
        }
        return reply, 200

class Benchmark(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()  

        #retrieve full trainLog
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, "trainLog": 1, 'gmmPickleStore':1})
        if db_result == None or 'trainLog' not in db_result: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        user_train_data =  extract_features(db_result['trainLog'])['train_data'] 
        user_gmm=pickle.loads(db_result['gmmPickleStore'])

        #get other user models 
        results = []
        for db_result in userData.find({'userId': {'$ne': JWT_userId}, 'trainProgress': {'$gte': 100}},{ '_id': 0, 'gmmPickleStore':1, 'scoreThreshold': 1, 'resultReference': 1, 'alias': 1, "trainLog": 1}, limit=10): 
            result={}            
            result["alias"]=db_result['alias']
            if 'gmmPickleStore' in db_result: #score user train data against alias model 
                alias_gmm=pickle.loads(db_result['gmmPickleStore'])
                score_array= alias_gmm.score_samples(user_train_data)
                result["result_alias_model"]=round(100*len(score_array[score_array>db_result['scoreThreshold']])/len(score_array))
            else: 
                result["result_alias_model"]=0
            if 'trainLog' in db_result:
                alias_train_data =  extract_features(db_result['trainLog'])['train_data']
                score_array= user_gmm.score_samples(alias_train_data)
                result["result_user_model"]=round(100*len(score_array[score_array>db_result['scoreThreshold']])/len(score_array))
            else:
                result["result_user_model"]=0
            results.append(result)
        reply = {
        'results' : results,
        }
        return reply, 200

class IntrusionTest(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()  

        #retrieve full trainLog
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, "trainLog": 1, 'gmmPickleStore':1, 'alias': 1, 'scoreThreshold': 1})
        if db_result == None or 'trainLog' not in db_result: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        user_gmm=pickle.loads(db_result['gmmPickleStore'])

        results = []
        #First result is reference result (for own model)
        result={}            
        result["alias"]=db_result['alias']+"(you!)"
        test_data =  extract_features(db_result['trainLog'])['test_data']
        score_array= user_gmm.score_samples(test_data)
        user_score_threshold=db_result['scoreThreshold']
        result["test_data_length"]=len(score_array)
        result["result"]=round(100*len(score_array[score_array>db_result['scoreThreshold']])/len(score_array))
        results.append(result)
        #get other user models 
        for db_result in userData.find({'userId': {'$ne': JWT_userId},'trainProgress': {'$gte': 100}},{ '_id': 0, 'alias': 1, "trainLog": 1}, limit=10): 
            result={}            
            result["alias"]=db_result['alias']
            if 'trainLog' in db_result:
                test_data =  extract_features(db_result['trainLog'])['test_data']
                score_array= user_gmm.score_samples(test_data)
                result["test_data_length"]=len(score_array)                
                result["result"]=round(100*len(score_array[score_array>user_score_threshold])/len(score_array))
            else:
                result["result"]=0
                result["test_data_length"]=0
            results.append(result)
        reply = {
        'results' : results,
        }
        return reply, 200




class Score(Resource):
    @jwt_required        
    def put(self):
        JWT_userId = get_jwt_identity()  
        JWT_token = get_raw_jwt()
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        #basic sanity check 
        if len(args['mfcc']) < 1 or len(args['energy']) < 1 or len(args['mfcc']) != len(args['energy']): 
            return {'error': 'data malformed - check the API specification'}, 400
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 1})
        if db_result == None: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        if (DISCARD_MFCC0):
            mfcc = np.delete(mfcc, 0, axis=1) #delete the 1st mfcc coeffecient 

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, 'gmmPickleStore':1, 'scoreThreshold': 1, 'resultReference': 1, 'scoreLog': 1})
        userData.update_one({'userId':JWT_userId},{'$unset': {'scoreLog': ""}})

        if db_result != None: 
            score_log_summary = [] #summary for API reply

            # load GMM model
            if 'gmmPickleStore' in db_result:
                gmm=pickle.loads(db_result['gmmPickleStore'])
                #score
                result = gmm.score_samples(mfcc)

                #get scoreLog and add new data 
                if 'scoreLog' in db_result:
                    score_log=db_result['scoreLog']
                else:
                    score_log=[]
                score_log_item = { 'timestamp': time(), 'session':JWT_token['jti'], 'data': request.json, 'result': round(100*len(result[result>db_result['scoreThreshold']])/len(result)) }
                score_log.append(score_log_item.copy())
                #trim if log too big 
                if len(score_log) > SCORELOGSIZE:
                    score_log.pop(0)

                score_log_summary_item = {}
                for log_entry in score_log:
                    score_log_summary_item['datetime'] = str(datetime.datetime.fromtimestamp(log_entry['timestamp']).isoformat())
                    score_log_summary_item['digit']= log_entry['data']['digit']
                    score_log_summary_item['result']= log_entry['result']
                    score_log_summary.append(score_log_summary_item.copy())

                reply = {
                    'average': np.average(result), 
                    'deviation': np.std(result), 
                    'min' : np.min(result),
                    'max' : np.max(result),
                    'length' : len(result),
                    'score' : len(result[result>db_result['scoreThreshold']]),
                    'scoring_log' : score_log_summary[::-1],
                    'result_reference' : db_result['resultReference'] # % of matched frames in test data set
                }
                #update log
                userData.update_one({'userId':JWT_userId},{'$set': {'scoreLog': score_log}})

            else: #no model trained 
                reply = {
                'average': 0, 
                'deviation': 0, 
                'min' : 0,
                'max' : 0,
                'length' : 0,
                'score' : 0,
                'scoring_log' : score_log_summary,
                'result_reference' : 0,
            }
            return reply, 200
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500

def extract_features(trainLog):
    #The trainLog contains a store of all training 'feature vectors' along with a 'label' of the digit spoken. 
    #The GMM model is to be re-built using the trainLog store split 50/50 for 'test data'/'train data'. 
        
    #Build list of mfcc arrays per spoken digit(0-9) from trainLog 
    digit_mfcc_list = [[],[],[],[],[],[],[],[],[],[]]
    for log_entry in trainLog:
        digit_mfcc_list[log_entry['data']['digit']].append(log_entry['data']['mfcc'])

    #Split odd/even indices for 'train data'/'test data' (per digit)
    train_mfcc_list = []
    test_mfcc_list =  []
    for digit_mfcc in digit_mfcc_list: 
        for idx, specific_digit in enumerate(digit_mfcc):
            if idx % 2:
                #even - train 
                if len(test_mfcc_list):
                    test_mfcc_list=np.append(test_mfcc_list,specific_digit,axis=0) 
                else:
                    test_mfcc_list=np.array(specific_digit)
            else:
                #odd - test
                if len(train_mfcc_list):
                    train_mfcc_list=np.append(train_mfcc_list,specific_digit,axis=0) 
                else:
                    train_mfcc_list=np.array(specific_digit)

    if (DISCARD_MFCC0):
        if len(train_mfcc_list):
            train_mfcc_list = np.delete(train_mfcc_list, 0, axis=1) #delete the 1st mfcc coeffecient 
        if len(test_mfcc_list):
            test_mfcc_list = np.delete(test_mfcc_list, 0, axis=1) #delete the 1st mfcc coeffecient 

    return { 'train_data': train_mfcc_list, 'test_data': test_mfcc_list}

#push data to templates 
@app.context_processor
def inject_user():
    return dict(GDPR_TEXT=cfg.GDPR_TEXT)

#serve static html templates
@app.route('/') 
def index():
    return render_template('index.html')
@app.route('/login') 
def session():
    return render_template('login.html')

#add API resources 
api.add_resource(ApiInfo, '/doc')
api.add_resource(Status, '/status')
api.add_resource(Delete, '/delete')
api.add_resource(Train, '/train')
api.add_resource(Benchmark, '/benchmark')
api.add_resource(IntrusionTest, '/intrusionTest')
api.add_resource(Score, '/score')
api.add_resource(Login, '/session')
api.add_resource(Register, '/user')

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

if __name__ == '__main__':
    app.run(debug=True)

