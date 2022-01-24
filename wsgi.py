import redis,yaml,json
from flask import Flask
from yaml.loader import SafeLoader


config = None
#TODO: implement waiting , now if the connections doesn't exist, the script will hang out indefinitely
with open('config.yaml') as f: # parsing config file
        config = yaml.load(f, Loader=SafeLoader)

conn = None
if config.get("socket") == None:
    if config.get("ip") == None:
        conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
    else:
      conn = redis.Redis(host=config["ip"]["host"], port=config["ip"]["port"], decode_responses=True)
else:
    conn = redis.Redis(unix_socket_path=config["socket"])


app = Flask(__name__)

@app.route('/')
def get_places():
    return conn.hgetall('parking')

        
