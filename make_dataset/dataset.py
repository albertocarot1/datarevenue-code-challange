import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional

import click
import pandas as pd
from category_encoders import OrdinalEncoder
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    flag = outdir / '.SUCCESS'

    out_train = outdir / 'train.csv'
    out_test = outdir / 'test.csv'

    train.to_csv(str(out_train), index=False)
    test.to_csv(str(out_test), index=False)

    flag.touch()


def split_dataset(df: DataFrame,
                  test_perc: int,
                  drop_duplicates: bool) -> Tuple[DataFrame, DataFrame]:
    """
    Split the dataset in train and test, in a stratified manner.

    Parameters
    ----------
    df: DataFrame
        dataset to be split
    test_perc: int
        percentage of test set
    drop_duplicates: bool
        whether duplicates should be dropped or not

    Returns
    -------
    train and test set

    """
    assert 1 < test_perc < 100
    # we set the index so we can properly execute loc below
    df = df.set_index('Unnamed: 0')

    if drop_duplicates:
        # remove possible duplicates
        df = df.drop_duplicates()

    # Get y so that train and test can be split equally
    y = df[['points']]

    test_size = test_perc / 100

    # split train and test stratified,
    # with fixed random seed for reproducibility.
    train, test, _, _ = train_test_split(df, y, test_size=test_size,
                                         random_state=42,
                                         stratify=y)
    return train, test


@dataclass
class Winery:
    """
    Dataclass to save a winery and its country, province and region_1
    """
    name: str
    country: Union[str, float] = float('nan')
    province: Union[str, float] = float('nan')
    region_1: Union[str, float] = float('nan')

    def is_fully_filled(self) -> bool:
        """
        Check that all the fields are full

        """
        if self.name and \
                str(self.country) != 'nan' and \
                str(self.province) != 'nan' and \
                str(self.region_1) != 'nan':
            return True
        return False


@dataclass
class Taster:
    """
    Data class to save a Taster, and the province (and corresponding country)
    where they make the most reviews.
    """
    name: str
    country: str
    province: str


class DatasetProcessor:
    """
    Class that processes the data from the wine rating problem.
    This is used to learn the preprocessing of the data (from a training set),
    to then apply it to a different dataset (e.g. a hold-out test set)
    The main methods are fit and transform. Transform cannot be called until
    fit has been executed.
    """
    def __init__(self, number_features: int = 100,
                 columns_to_remove: list = None):

        # Columns to remove from initial dataframe
        if not columns_to_remove:
            self.columns_to_remove: list = ['description', 'designation',
                                            'title', 'winery']

        # Number of text features to extract through tfidf
        self.number_features: int = number_features

        # Value
        self.year_fill_value: Optional[float] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.tasters_twitter: Optional[Dict[str, set]] = None
        self.twitters_taster: Optional[Dict[str, set]] = None
        self.price_fill_value: Optional[float] = None
        self.wineries: Optional[Dict[str, Winery]] = None
        self.tasters: Optional[Dict[str, Taster]] = None
        self.encoder: Optional[OrdinalEncoder] = None

    @staticmethod
    def _extract_years_from_dataframe(df: DataFrame) -> Series:
        """
        Extract the year of all wines from the titles.

        Parameters
        ----------
        df: DataFrame
            dataframe with the column year

        Returns
        -------
            years: Series
                A Series with all the years for each row (including missing)
        """
        years = [float('nan') for _ in range(len(df['title']))]
        for i, title in enumerate(df['title']):
            # Find 4 consecutive digits in the title's text
            matched_years = re.findall(r"\d{4}", title)
            for year in matched_years:
                # Only use numbers that might actually be wine years
                if year[:2] in ['18', '19', '20']:
                    years[i] = float(year)
        return Series(years)

    @staticmethod
    def _extract_twitters_tasters(df: DataFrame) -> \
            Tuple[Dict[Any, set], Dict[Any, set]]:
        """
        Create two dictionaries, one with all the twitter accounts given a
        certain Taster, the other with all the tasters given a certain twitter
        account.

        Parameters
        ----------
        df: DataFrame
            Dataframe with `taster_name` and `taster_twitter_handle` columns

        Returns
        -------
        The two symmetrical dictionaries of tasters and their twitter account

        """
        tasters_twitter = {}
        twitters_taster = {}

        for name, twitter in zip(df["taster_name"],
                                 df["taster_twitter_handle"]):
            # Create a dict with a set containing all the
            # twitter accounts connected to a name
            if not tasters_twitter.get(name):
                tasters_twitter[name] = {twitter}
            else:
                tasters_twitter[name].add(twitter)
            # Create a dict with a set containing all the
            # names connected to a twitter account
            if not twitters_taster.get(twitter):
                twitters_taster[twitter] = {name}
            else:
                twitters_taster[twitter].add(name)
        return tasters_twitter, twitters_taster

    def _fit_learn_features_from_description(self, train_df):
        """
        Train TfIdf model with descriptions' texts

        Parameters
        ----------
        train_df: DataFrame
            Dataframe from which to extract the descriptions

        Returns
        -------
        Trained tfidf vectorizer
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                           ngram_range=(1, 1),
                                           lowercase=True,
                                           max_features=self.number_features)

        tfidf_vectorizer.fit(train_df['description'])
        print("Features extracted from training set description are:")
        print(tfidf_vectorizer.get_feature_names())
        return tfidf_vectorizer

    @staticmethod
    def _fit_extract_wineries(train_df: DataFrame) -> Dict[str, Winery]:
        """
        Create a dict with winery name as key, and `Winery` object as value.
        Winery's province, country and region, are taken from the first review
        found that contains all of them. This might not exist.

        Parameters
        ----------
        train_df: DataFrame
            Dataframe with `winery`, `country`, `province` and `region_1` among
            the columns.

        Returns
        -------
            wineries: Dict
                dict with winery name as key, and `Winery` as value
        """
        wineries = {}

        for winery in set(train_df['winery']):  # iterate through all wineries
            # Iterate reviews from the same winery
            for i, (_, review) in enumerate(train_df[train_df['winery'] == winery].iterrows()):
                if i == 0:
                    # Always save the info from the first review found
                    wineries[winery] = Winery(winery,
                                              review['country'],
                                              review['province'],
                                              review['region_1'])
                elif not wineries[winery].is_fully_filled():
                    # Update the winery data if some info are still missing
                    wineries[winery] = Winery(winery,
                                              review['country'],
                                              review['province'],
                                              review['region_1'])
                else:
                    break
        return wineries

    @staticmethod
    def _fit_get_taster_favourite_area(train_df):
        """
        Create a dict with all tasters, with taster name as key, and a `Taster`
        object as value. The object will be filled with the taster's
        country and province he did the most reviews of.

        Parameters
        ----------
        train_df: DataFrame
            Dataframe with columns `taster_name`, `country`, `province`

        Returns
        -------
            tasters: dict
                Dict with tasters names as keys, and Taster object as value
        """
        tasters: dict = {}
        for taster in set(train_df['taster_name']):
            if not isinstance(taster, str) and math.isnan(taster):
                continue
            # Take all the taster's reviews that have country and province
            reviews_same_taster = train_df[train_df['taster_name'] == taster][
                (train_df['country'].notna()) & (train_df['province'].notna())]
            # Count the number of reviews for each province
            reviews_per_province = reviews_same_taster.groupby(['province'])[
                'country'].count()
            # Identify province with most reviews
            most_reviews_single_province = max(reviews_per_province)
            for province, count in reviews_per_province.to_dict().items():
                if count == most_reviews_single_province:
                    # Assume the same province is from the same country
                    country = reviews_same_taster[
                        reviews_same_taster["province"] == province][
                        "country"].iloc[0]
                    tasters[taster] = Taster(taster,
                                             country,
                                             province)
                    break
        return tasters

    @staticmethod
    def _fit_learn_categorical_variables(train_df: DataFrame):
        """
        Train encoder to transform categorical variables in numerical ones.

        Parameters
        ----------
        train_df: DataFrame
            dataframe with categorical variables

        Returns
        -------
            encoder: OrdinalEncoder
                encoder to change the
        """

        cols_to_encode = ['country', 'province', 'region_1',
                          'region_2', 'taster_name', 'taster_twitter_handle',
                          'variety']

        encoder = OrdinalEncoder(cols=cols_to_encode, return_df=True)

        # Fit Data
        encoder.fit(train_df)

        return encoder

    def fit(self, train_df: DataFrame):
        """
        For all transformations based on data, use the training set to
        learn how to transform data. For instance, if in the test set there
        are values for a column that are missing from the training set, it
        should be handled by assigning a fixed value, instead of throwing
        an error.

        Parameters
        ----------
        train_df: DataFrame
            Dataframe to train the processor

        """

        self.year_fill_value: float = self._extract_years_from_dataframe(train_df).median()
        self.tfidf: TfidfVectorizer = self._fit_learn_features_from_description(train_df)
        self.tasters_twitter, self.twitters_taster = self._extract_twitters_tasters(
            train_df)

        self.price_fill_value = train_df["price"].median()
        self.wineries = self._fit_extract_wineries(train_df)
        self.tasters = self._fit_get_taster_favourite_area(train_df)

        self.encoder = self._fit_learn_categorical_variables(train_df)

    def _transform_extract_features_from_description(self,
                                                     df_to_transform: DataFrame):
        """
        Extract textual features from the description column

        Parameters
        ----------
        df_to_transform: DataFrame
            Data frame that needs to be transformed

        Returns
        -------
            transformed_df: DataFrame
                Dataframe with additional text features
        """
        description_features = self.tfidf.transform(df_to_transform['description'])
        dense_features = pd.DataFrame(description_features.toarray())
        dense_features.columns = [f"desc_{name}" for name in
                                  self.tfidf.get_feature_names()]
        df_to_transform = df_to_transform.reset_index()
        df_to_transform = df_to_transform.drop(columns=['Unnamed: 0'])
        df_to_transform = pd.concat([df_to_transform, dense_features], axis=1)
        return df_to_transform

    def _transform_update_twitter_name(self, df_to_transform):
        """
        Replace the twitter profile for tasters that have the same profile
        in every single wine review.

        Parameters
        ----------
        df_to_transform: DataFrame
            Dataframe that needs to be transformed.

        """
        tasters_twitter, twitters_taster = self._extract_twitters_tasters(
            df_to_transform)

        for name, twitter_set in tasters_twitter.items():

            # Skip missing names
            if str(name) == "nan":
                continue

            # Assume writers have no more than one twitter
            twitter = str(list(twitter_set)[0])

            # Skip missing twitter
            if str(twitter) == "nan":
                continue

            if len(self.twitters_taster[twitter]) == 1:
                df_to_transform.loc[df_to_transform["taster_name"] == name,
                                    "taster_twitter_handle"] = "personal"

    def _transform_add_year(self, df_to_transform):
        """
        Add the year column (by extracting it from the title), and fill the
        empty values.

        Parameters
        ----------
        df_to_transform: DataFrame
            Dataframe with a column `title`
        """
        df_to_transform["year"] = self._extract_years_from_dataframe(
            df_to_transform)
        df_to_transform["year"] = df_to_transform["year"].fillna(
            self.year_fill_value)

    def _transform_fill_location_columns(self, df_to_transform: DataFrame):
        """
        Fill country, province and region_1, making a best effort to infer them
        from the winery, and then from the taster.

        Parameters
        ----------
        df_to_transform: DataFrame
            Dataframe with the columns `country`, `winery`, `province`,
            `region_1` and `taster_name`
        """
        c1 = df_to_transform['country'].isna()
        c2 = df_to_transform['winery'].notna()
        for winery in df_to_transform[c1 & c2]['winery']:
            c1 = df_to_transform['country'].isna()
            c2 = df_to_transform['winery'] == winery

            df_to_transform.loc[c1 & c2, "country"] = self.wineries[
                winery].country

            c3 = df_to_transform['province'].isna()

            df_to_transform.loc[c3 & c2, "province"] = self.wineries[
                winery].province

            c4 = df_to_transform['region_1'].isna()

            df_to_transform.loc[c4 & c2, "region_1"] = self.wineries[
                winery].region_1

        c1 = df_to_transform['country'].isna()
        c2 = df_to_transform['taster_name'].notna()
        for taster in df_to_transform[c1 & c2]['taster_name']:
            c1 = df_to_transform['country'].isna()
            c2 = df_to_transform['taster_name'] == taster
            c3 = df_to_transform['province'].isna()

            df_to_transform.loc[((c1 | c3) & c2), "province"] = \
                self.tasters[taster].province

            df_to_transform.loc[((c1 | c3) & c2), "country"] = self.tasters[
                taster].country

    def transform(self, df_to_transform: DataFrame):
        """
        Transforms the columns of the dataset, returning a dataframe ready
        to be passed on to the model for inference or training.

        Parameters
        ----------
        df_to_transform: DataFrame
            dataframe to transform

        Returns
        -------
            transformed_df: DataFrame
                Transformed dataframe
        """

        assert self.tfidf is not None, ".fit() must be launched before .transform"
        self._transform_update_twitter_name(df_to_transform)
        self._transform_fill_location_columns(df_to_transform)

        df_to_transform["price"] = df_to_transform["price"].fillna(
            self.price_fill_value)
        # Since region_2 is specified only for wines with US as country
        df_to_transform['region_2'] = df_to_transform['region_2'].fillna(
            "not_US")
        df_to_transform['region_1'] = df_to_transform['region_1'].fillna(
            "unknown")
        df_to_transform['taster_name'] = df_to_transform['taster_name'].fillna(
            "unknown")
        df_to_transform['taster_twitter_handle'] = df_to_transform[
            'taster_twitter_handle'].fillna("missing")
        df_to_transform = self.encoder.transform(df_to_transform)
        self._transform_add_year(df_to_transform)
        df_to_transform = self._transform_extract_features_from_description(df_to_transform)
        df_to_transform = df_to_transform.drop(columns=self.columns_to_remove)
        return df_to_transform


@click.command()
@click.option('--in-csv')
@click.option('--test-perc', type=int)
@click.option('--out-dir')
@click.option('--drop-duplicates')
def make_datasets(in_csv, test_perc, out_dir, drop_duplicates):
    assert drop_duplicates in ['true', 'false']
    drop_duplicates = True if drop_duplicates == 'true' else False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    train, test = split_dataset(df, test_perc, drop_duplicates)
    data_processor = DatasetProcessor()
    data_processor.fit(train)
    train = data_processor.transform(train)
    test = data_processor.transform(test)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
