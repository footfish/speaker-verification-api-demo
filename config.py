import os

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")

if not JWT_SECRET_KEY:
    raise ValueError("No JWT_SECRET_KEY set for application")
if not MONGODB_URI:
    raise ValueError("No MONGODB_URI set for application")
