import yaml, json
import pandas as pd

# ----- Read file -----
def yaml_read(dir): 
    content = yaml.load(open(dir, 'r'), Loader=yaml.FullLoader)
    return content

def json_load(dir): 
    content = json.load(open(dir, 'r'))
    return content

def json_dump(data, dir):
    json.dump(data, open(dir, 'w')) 
