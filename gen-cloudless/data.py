import os
from sklearn.model_selection import train_test_split


def get_file_ids():
    base = 'ROIs1868_summer'
    postfixes = ['s1', 's2', 's2_cloudy']

    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # get the data directory
    data_dir = os.path.abspath(os.path.join(current_dir, "data"))

    # get all files in the dirs
    files = []
    for postfix in postfixes:
        d = os.path.join(data_dir, f'{base}_{postfix}', "")
        for id in os.listdir(d):
            filedir = os.path.join(d, id)
            for f in os.listdir(filedir):
                without_base = f.replace(base, '')
                without_prefix = without_base.replace(f'_{postfix}_', '')
                files.append(without_prefix)

    files = set(files)
    return files


def get_train_val_test_split(train_size=0.8):
    file_ids = list(get_file_ids())

    train_ids, temp_ids = train_test_split(file_ids, train_size=train_size)
    val_ids, test_ids = train_test_split(temp_ids, train_size=.5)

    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    t, v, te = get_train_val_test_split()
    print(len(t), len(v), len(te))
