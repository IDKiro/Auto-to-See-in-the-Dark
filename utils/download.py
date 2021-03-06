import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

res = input("Download Model? [y/n]: ")
while res is not 'y' and res is not 'n' and res is not 'Y' and res is not 'N':
    res = input("[Input error] Download Model? [y/n]: ")
if res is 'y' or res is 'Y':
    print('Downloading Model... (89MB)')
    download_file_from_google_drive('1xdw4oJbrfeV5U6YZMDCIXJxjIIa8fOhE', 'checkpoint/model.ckpt.data-00000-of-00001')
    download_file_from_google_drive('1Pv0y04lDO0CfhGVQ6IbeeMZFysF7QRGU', 'checkpoint/model.ckpt.meta')


res = input("Download Dataset? [y/n]: ")
while res is not 'y' and res is not 'n' and res is not 'Y' and res is not 'N':
    res = input("[Input error] Download Dataset? [y/n]: ")
if res is 'y' or res is 'Y':
    print('Downloading Dataset... (25GB)')
    download_file_from_google_drive('10kpAcvldtcb9G2ze5hTcF1odzu4V_Zvh', 'dataset/Sony.zip')
    os.system('unzip dataset/Sony.zip -d dataset')
