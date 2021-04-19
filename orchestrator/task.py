import os
from pathlib import Path

import luigi

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
            path=str(out_dir / f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):
    """
    Pipeline task that splits the dataset in train and test,
    and processes it for training and evaluation.
    """

    in_csv = luigi.Parameter(default="/usr/share/data/raw/wine_dataset.csv")
    test_percentage = luigi.Parameter(default="20")
    out_dir = luigi.Parameter(default="/usr/share/data/processed/")
    drop_duplicates = luigi.Parameter(default='--drop-duplicates')

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
            self.drop_duplicates
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModel(DockerTask):
    """
    This task trains the model with the previously created train set, and saves
    it in a file.
    """

    train_set_path = luigi.Parameter(
        default="/usr/share/data/processed/train.csv")
    model_out_file = luigi.Parameter(
        default="/usr/share/data/models/xgbr.model")

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        completed_task = MakeDatasets()
        datasets_folder = os.path.dirname(completed_task.output().path)
        self.train_set_path = os.path.join(datasets_folder, 'train.csv')
        return completed_task

    @property
    def command(self):
        return [
            'python', 'train_model.py',
            '--train-set-path', self.train_set_path,
            '--model-out-file', self.model_out_file
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.model_out_file))
        )


class EvaluateModel(DockerTask):
    # evaluates the model, and creates a report notebooks HTML export file.
    # Write in README documentation where the report will be found
    test_set_path = luigi.Parameter(default="/usr/share/data/processed/test.csv")
    model_file = luigi.Parameter(default="/usr/share/data/models/xgbr.model")


    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        completed_task = TrainModel()
        self.model_file = completed_task.output()
        return completed_task

    @property
    def command(self):
        return [
            'python', 'evaluate_model.py',
            '--test-set-path', self.test_set_path,
            '--model-file', self.model_file,
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )
