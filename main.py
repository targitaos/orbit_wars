import os

import kagglehub

if __name__ == '__main__':
    # kagglehub.login()

    # Download latest version
    path = kagglehub.competition_download('orbit-wars')

    print('Path to competition files:', path)
