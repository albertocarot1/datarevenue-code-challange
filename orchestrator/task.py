import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2020/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):
    """ Pipeline task that splits the dataset in train and test. """

    in_csv = luigi.Parameter(default="/usr/share/data/raw/wine_dataset.csv")
    test_percentage = luigi.Parameter(default="20")
    out_dir = luigi.Parameter(default="/usr/share/data/processed/")
    drop_duplicates = luigi.Parameter(default='true')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        completed_task = DownloadData()
        self.in_csv = completed_task.output().path
        return completed_task

    @property
    def command(self):
        return [
            'python', 'dataset.py',
            '--in-csv', self.in_csv,
            '--test-perc', self.test_percentage,
            '--out-dir', self.out_dir,
            '--drop-duplicates', self.drop_duplicates
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

# class TrainModel(DockerTask):
#     # TODO execute here click script that trains the model, and saves it to
#     # a binary file. Indicate as output, the location of the model file (or folder)
#
#     in_csv = luigi.Parameter()
#     test_percentage = luigi.Parameter(default=30)
#     out_dir = luigi.Parameter(default="/usr/share/data/split/")
#
#
#     @property
#     def image(self):
#         return f'code-challenge/make-dataset:{VERSION}'
#
#     def requires(self):
#         self.in_csv = DownloadData()
#         return self.in_csv
#
#     @property
#     def command(self):
#         # TODO: implement correct command
#         # Try to get the input path from self.requires() ;)
#         return [
#             'python', 'dataset.py',
#             '--in-csv', self.in_csv,
#             '--test-perc', self.test_percentage,
#             '--out-dir', self.out_dir
#         ]
#
#     def output(self):
#         return luigi.LocalTarget(
#             path=str(Path(self.out_dir) / '.SUCCESS')
#         )
#
# class EvaluateModel(DockerTask):
#     # TODO execute here click script that evaluates the model, and creates
#     # a report through notebooks. Write in README documentation where the report
#     # will be found
#     in_csv = luigi.Parameter()
#     test_percentage = luigi.Parameter(default=30)
#     out_dir = luigi.Parameter(default="/usr/share/data/split/")
#
#
#     @property
#     def image(self):
#         return f'code-challenge/make-dataset:{VERSION}'
#
#     def requires(self):
#         self.in_csv = DownloadData()
#         return self.in_csv
#
#     @property
#     def command(self):
#         # TODO: implement correct command
#         # Try to get the input path from self.requires() ;)
#         return [
#             'python', 'dataset.py',
#             '--in-csv', self.in_csv,
#             '--test-perc', self.test_percentage,
#             '--out-dir', self.out_dir
#         ]
#
#     def output(self):
#         return luigi.LocalTarget(
#             path=str(Path(self.out_dir) / '.SUCCESS')
#         )