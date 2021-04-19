import logging
import os
import subprocess

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('evaluate-model')


@click.command()
@click.option('--test-set-path',
              required=True,
              help='Path to the test set csv file')
@click.option('--model-file',
              required=True,
              help='Path where the trained model file can be found')
@click.option('--report-dir',
              required=True,
              help='Path where the report will be saved')
def evaluate_model(test_set_path, model_file, report_dir):
    """
    Create a report with evaluation of the model on the test set.
    The report is the export of the notebook EvaluateModel.ipynb in html format.
    """

    assert '.csv' in test_set_path, f'Received {test_set_path}! ' \
                                    f'Please provide a .csv file'
    logger.info("Re-running notebook")
    def generate_arguments(args_dict):
        """Create a 'arguments.py' module to initialize a Jupyter notebook."""
        with open('arguments.py', 'w') as open_pyfile:
            for key, value in args_dict.items():
                open_pyfile.write(f'{key} = {repr(value)}\n')
    logger.info("Generating file with variables")
    # Prepare the arguments
    generate_arguments({
        'test_set_path': test_set_path,
        'model_file': model_file
    })
    os.makedirs(os.path.dirname(report_dir), exist_ok=True)
    logger.info("Launching nbconvert")
    p = subprocess.Popen(['jupyter-nbconvert',
                          '--execute',
                          '--output-dir', report_dir,
                          '--to', 'html',
                          'EvaluateModel.ipynb'])

    while p.poll() is None:
        continue

    logger.info(f'Notebook has been exported as .html in {report_dir}')


if __name__ == '__main__':
    evaluate_model()
