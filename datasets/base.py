from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def load_movie_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        movie_df = self.load_movie_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        movie_genre_df = self.get_movie_genre_df(movie_df, smap)
        df = pd.merge(df, movie_genre_df, how='inner', on='sid')
        train, val, test = self.split_df(df, len(umap))
        movie_genre_map = {sid: genre_index for sid, genre_index in zip(movie_genre_df.sid, movie_genre_df.genre_index)}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'movie_genre_map': movie_genre_map}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def get_movie_genre_df(self, movie_df, smap):
        # Converting genre to numerical ids.
        genre_id_map = {u: i for i, u in enumerate(set(movie_df['genres']))}
        movie_df['genre_index'] = movie_df['genres'].map(genre_id_map)
        # Change the sid to densified sid.
        # Step1. Convert smap to a dataframe.
        old_id = list(smap.keys())
        new_id = list(smap.values())
        new_smap = {'sid': old_id, 'new_sid': new_id}
        smap_df = pd.DataFrame.from_dict(new_smap)
        # Step2. Join with SMAP to attain new movie ids.
        movie_df = pd.merge(movie_df, smap_df, how='inner', on='sid')
        # Step3. Dropping unnecessary columns and forming final df.
        movie_df.drop(['sid', 'title', 'genres'], axis=1, inplace=True)
        movie_df.columns = ['genre_index', 'sid']
        return movie_df

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(zip(d.sort_values(by='timestamp')['sid'],
                                                                      d.sort_values(by='timestamp')['genre_index'])))

            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(zip(d['sid'], d['genre_index']))))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(zip(d['sid'], d['genre_index']))))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(zip(d['sid'], d['genre_index']))))

            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

