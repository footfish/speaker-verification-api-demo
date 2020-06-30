import config as cfg
import numpy as np
from flask import Flask, abort, render_template, request
from flask_restful import reqparse,Resource, Api
from flask_jwt_extended import create_access_token,JWTManager,jwt_required, get_jwt_identity, get_raw_jwt, get_jti
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

n_gmmcomponents=5
discard_mfcc0=True #choose to discard first cepstral coefficient 

parser = reqparse.RequestParser()
parser.add_argument('mfcc', type=list, location='json')
parser.add_argument('energy', type=list, location='json')
parser.add_argument('username')
parser.add_argument('alias')

# Provide api information
class ApiInfo(Resource):
    def get(self):
        return {'version': '0.1',
                'usage': '/doc /user /login /status /train /test /score /delete'
                }

# 'Login user' and return JWT
class Login(Resource):
  def post(self):
    args = parser.parse_args()
    user = userData.find_one({"email": args['username']})
    if user != None:
        print(args['username'] + ' logged in')
        expires = datetime.timedelta(days=7)
        access_token = create_access_token(identity=user['userId'], expires_delta=expires)
        return {'token': access_token}, 200
    else:
        return {'error': 'Email invalid'}, 401

# 'Register user' and return JWT
class Register(Resource):
  def post(self):
    args = parser.parse_args()
    # TODO validate 

    print(args['username'] + ' registering')
    user = userData.find_one({"email": args['username']})

    if user == None:
        userData.insert_one({ "email": args['username'], "userId":hash(args['username']), "alias": args['alias'] })
        print(args['username'] + ' registered')
        expires = datetime.timedelta(days=7)
        access_token = create_access_token(identity=hash(args['username']), expires_delta=expires)
        return {'token': access_token}, 200
    else:
        return {'error': 'Already exists'}, 403


# Status: Deletes trainLog store & GMM pickle & resets stats. 
class Delete(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        userData.update_one({'userId':JWT_userId},{'$unset': {'gmmPickleStore': ""}})
        userData.update_one({'userId':JWT_userId},{'$unset': {'trainLog': ""}})
        userData.update_one({'userId':JWT_userId},{'$set': {'trainDataLength': 0, 'testDataLength': 0, 'trainProgress': 0}})
        reply = {
        'deleted' : True,    
        }
        return reply, 200

# Status: Returns length of test and train store. 
class Status(Resource):
    @jwt_required        
    def get(self):
        JWT_userId = get_jwt_identity()
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0,'trainDataLength':1,'testDataLength':1, 'trainProgress':1 })
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
            reply = {
            'train_data_length' : train_len,
            'test_data_length' : test_len,
            'training_progress' : train_prog,
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
        mfcc = np.array(args['mfcc'])     
        #basic sanity check 
        if len(args['mfcc']) < 1 or len(args['energy']) < 1 or len(args['mfcc']) != len(args['energy']): 
            return {'error': 'data malformed - check the API specification'}, 400

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 1})
        if db_result == None: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        #log request to trainLog
        trainLog = { 'timestamp': time(), 'session':JWT_token['jti'], 'data': request.json }
        userData.update_one({'userId':JWT_userId},{'$push': {'trainLog': trainLog}})

        #retrieve full trainLog
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, "trainLog": 1})

        #The trainLog contains a store of all training 'feature vectors'along with a 'label' of the digit spoken. 
        #The GMM model is to be re-built using the trainLog store split 50/50 for 'test data'/'train data'. 

        
        #Build list of mfcc arrays per spoken digit(0-9) from trainLog 
        digit_mfcc_list = [[],[],[],[],[],[],[],[],[],[]]
        for log_entry in db_result['trainLog']:
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

        #calculate training progress 
        api_training_progress = int(100*len(db_result['trainLog'])/60)  #expect 60 digits to be recorded. 

        #build summary trainlog returned in api 
        api_log_list = []
        api_log_entry = {}
        for log_entry in db_result['trainLog']:
            api_log_entry['datetime'] = str(datetime.datetime.fromtimestamp(log_entry['timestamp']).isoformat())
            api_log_entry['digit']= log_entry['data']['digit']
            api_log_list.append(api_log_entry.copy())

        if (discard_mfcc0):
            if len(train_mfcc_list):
                train_mfcc_list = np.delete(train_mfcc_list, 0, axis=1) #delete the 1st mfcc coeffecient 
            if len(test_mfcc_list):
                test_mfcc_list = np.delete(test_mfcc_list, 0, axis=1) #delete the 1st mfcc coeffecient 

        # create model
        gmm = mixture.GaussianMixture(n_gmmcomponents, covariance_type='full')

        # TODO/check 
        #prune mfcc file if too large         
        #fit (warm start may help)

        tTime=time()
        gmm.fit(train_mfcc_list)
        tTime=time()-tTime

        #score 
        if len(test_mfcc_list):
            testResult = gmm.score_samples(test_mfcc_list)
            scoreThreshold=np.average(testResult)-np.std(testResult)
            resultReference=round(100*len(testResult[testResult>scoreThreshold])/len(testResult)) # % of matched frames in test data set
        else:
            scoreThreshold=0
            resultReference=0
            

        #Update db - gmm pickle / cache stats 
        userData.update_one({'userId':JWT_userId},{'$set': {'gmmPickleStore': Binary(pickle.dumps(gmm)), 'resultReference': resultReference,'scoreThreshold': scoreThreshold,'trainDataLength': len(train_mfcc_list), 'testDataLength': len(test_mfcc_list), 'trainProgress': api_training_progress}})

        reply = {
        'training_progress' : api_training_progress,
        'train_data_length' : len(train_mfcc_list),
        'test_data_length' : len(test_mfcc_list),
        'training_time' : tTime,
        'training_log' : api_log_list,
        'score_threshold': scoreThreshold, 
        'result_reference' : resultReference
        }
        return reply, 200


class Score(Resource):
    @jwt_required        
    def put(self):
        JWT_userId = get_jwt_identity()  
        #JWT_token = get_raw_jwt()
        #read mfcc features from resource
        args = parser.parse_args()
        mfcc = np.array(args['mfcc'])        
        #basic sanity check 
        if len(args['mfcc']) < 1 or len(args['energy']) < 1 or len(args['mfcc']) != len(args['energy']): 
            return {'error': 'data malformed - check the API specification'}, 400
        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 1})
        if db_result == None: 
            return {'error': 'this is unexpected, can not find user data'}, 500

        if (discard_mfcc0):
            mfcc = np.delete(mfcc, 0, axis=1) #delete the 1st mfcc coeffecient 

        #log 
#        scoreLog = { 'timestamp': time(), 'session':JWT_token['jti'], 'data': request.json }
#        userData.update_one({'userId':JWT_userId},{'$push': {'scoreLog': scoreLog}})

        db_result = userData.find_one({'userId':JWT_userId},{ "_id": 0, 'gmmPickleStore':1, 'scoreThreshold': 1, 'resultReference': 1})
        if db_result != None: 
            # load GMM model
            if 'gmmPickleStore' in db_result:
                gmm=pickle.loads(db_result['gmmPickleStore'])
                #score 
                result = gmm.score_samples(mfcc)
                reply = {
                    'average': np.average(result), 
                    'deviation': np.std(result), 
                    'min' : np.min(result),
                    'max' : np.max(result),
                    'length' : len(result),
                    'score' : len(result[result>db_result['scoreThreshold']]),
                    'result_reference' : db_result['resultReference'] # % of matched frames in test data set
                }
            else: #no model trained 
                reply = {
                'average': 0, 
                'deviation': 0, 
                'min' : 0,
                'max' : 0,
                'length' : 0,
                'score' : 0,
                'result_reference' : 0,
            }
            return reply, 200
        else:
            return {'error': 'this is unexpected, can not find user data'}, 500

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

