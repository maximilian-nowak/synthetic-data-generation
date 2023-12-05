# MIT License
#
# Copyright (c) 2023 Eduard Bartolovic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""CLI python script.
Typical usage example:
  python run.py --source samples/ --config configs/config_detection.yaml
"""
import argparse
import logging
import os

from src.datageneration import pipeline


def main(opt):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    cwd = os.getcwd()
    pipeline.pipeline(opt, cwd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, help="path to the images source")
    parser.add_argument("-t", "--target", default="download", help="path to the target dir")
    parser.add_argument("-c", "--config", required=True, help="path to the config file")

    parsed_opt = parser.parse_args()
    main(parsed_opt)
