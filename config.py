import os

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
GDPR_TEXT = '''
Please read the data and privacy statement in full:
-------------------------------------------------------
This web application has been created for research purposes. You must be aged 16 or over to register and contribute. 
By registering with this application and checking the consent boxes below you are acknowledging that all data collected and stored is contributed as open data, with the exception of user credentials (email/username). 

All data collected (with the exception of user credentials) is contributed with a Creative Commons Attribution (CC-BY) Licence - https://creativecommons.org/licenses/by/4.0/legalcode 

The following user data is collected and stored: 
- 'User credential' data (email/username) * NOTE: This is private, never shared 
- 'Alias' data collected during registration. 
- 'Speech feature vector' data. ** NOTE: This is classified as personal biometric data, as such it may possibly be used to identify you. 
- 'Demographic' data collected on the form during registration (gender, age group)
- 'Performance log' data collected and calculated during speech training and scoring. 
- 'Browser information' data collected from browser headers during session login. 

Why data is collected and stored: 
- 'User credential' data (email/username) is used to group and log your sessions only. 
- 'Alias' is data publicly shared to identify your performance logs. 
- 'Speech feature vector', 'Demographic', 'Browser information', and 'Performance log' data is contributed as open data for research purposes.  

Where data is stored: 
  With the exception of user credentials, all other data may be archived and made publicly available and stored anywhere under the Creative Commons Attribution (CC-BY) Licence. 
  Live application user data is stored in a MongoDB hosted in an mlab sandbox - https://www.mlab.com/
  Mlab sandbox is utilising Amazon Web Services hosting location US East (N. Virginia)
  mLab has implemented the necessary practices and documentation to address the requirements of GDPR
  Full details of this can be found here https://docs.mlab.com/eu-data-protection/
  
How long will it will be stored for: 
- 'User credential' (ie. email) data will be obfiscated regularly. I suggest using a made up email address.
- 'Other' data may be stored indefinitely. 
'''

if not JWT_SECRET_KEY:
    raise ValueError("No JWT_SECRET_KEY set for application")
if not MONGODB_URI:
    raise ValueError("No MONGODB_URI set for application")
