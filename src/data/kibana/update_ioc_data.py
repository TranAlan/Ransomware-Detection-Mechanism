#!/usr/bin/python
'''This python script will update the existing ioc list to add more values for Kibana.'''

import json
import os
import sys
import requests
import json_translator as translator
sys.path.append('../')
import utils.file_util as util

VT_API_PARAM = 'apikey'
VT_HASH_PARAM = 'hash'
API_ENV_VAR = 'RDM_API_KEY'


def main():
    '''main'''
    config = util.load_yaml(
        'C:/Users/Owner/Documents/Git-Repos/Ransomware-Detection-Mechanism/src/data/config.yml')
    vt_url = "https://www.virustotal.com/vtapi/v2/file/report"
    api_key = '8911156fb712ee8e02e68029bcf8ff2d8aed94d24c2c0a3638ba32ae22941f09'

    hash_list, ip_list, url_list = get_hashes()
    # for value in range(0, 1):
    #     resource += (hash_list[value])
    #     if value != (len(hash_list) -1):
    #         resource += ", "
    resource = "{}, {}, {}, {}, {}, {}".format(hash_list[1],hash_list[2],hash_list[3],hash_list[4],hash_list[5],hash_list[6])


    params = {'apikey': api_key, 'resource': resource}
    # print(os.environ.get(API_ENV_VAR))
    response = requests.get(vt_url, params=params)
    # print(response)
    print(response.json())
    print(len(response.json()))
    # total = response.json()["total"]
    # positives = response.json()["positives"]
    # percent = round((positives/total)*100, 2)
    # scan_date = response.json()["scan_date"]
    # print (percent)




def get_hashes():
    '''Reads a JSON file containing a list of hashes'''
    ioc_list = translator.convert_to_json(
        'C:/Users/Owner/Documents/Git-Repos/Ransomware-Detection-Mechanism/src/data/inputs/ioc_list.json')
    hash_list = []
    ip_list = []
    url_list = []
    for ioc in ioc_list:
        if (json.loads(ioc))["type"] == "MD5" or (json.loads(ioc))["type"] == "SHA256":
            hash_list.append((json.loads(ioc))["value"])
        if (json.loads(ioc))["type"] == "IP":
            ip_list.append((json.loads(ioc))["value"])
        if (json.loads(ioc))["type"] == "URL":
            url_list.append((json.loads(ioc))["value"])
    return hash_list, ip_list, url_list


if __name__ == '__main__':
    main()
