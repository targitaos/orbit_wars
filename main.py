import os

import kagglehub

if __name__ == '__main__':
    kagglehub.login(api_token=os.environ['KAGGLE_API_TOKEN'])

    # Download latest version
    path = kagglehub.competition_download('orbit-wars')

    print('Path to competition files:', path)
