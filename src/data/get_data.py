import kaggle
import zipfile


def load_competition_data(path, dataname):
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(dataname, path=path)
    unzip_data(path, dataname + '.zip')


def unzip_data(path, file):
    with zipfile.ZipFile(path + file, 'r') as zip_ref:
        zip_ref.extractall(path)


if __name__ == "__main__":
    load_competition_data('./data/raw/', 'widsdatathon2021')
