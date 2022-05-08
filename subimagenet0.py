# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imagenet datasets."""

import io
import os
import tarfile

from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """SubImageNet with 100 classes according to https://github.com/sud0301/essentials_for_CIL/
blob/main/data/imagenet100_s1993.txt"""

label_list = ["n02123159", "n02124075", "n03476991", "n03843555", "n02110627", "n03720891", "n04417672", "n02423022",
              "n03977966", "n01629819", "n02100236", "n03379051", "n02074367", "n03584254", "n02167151", "n01877812",
              "n02981792", "n04238763", "n02951358", "n01855672", "n03444034", "n02097298", "n02979186", "n03976467",
              "n01630670", "n04370456", "n01704323", "n01930112", "n02107683", "n04310018", "n03759954", "n02948072",
              "n01558993", "n02490219", "n02264363", "n04465501", "n03584829", "n04507155", "n12998815", "n02110958",
              "n02667093", "n02088238", "n02791270", "n03776460", "n02106030", "n07754684", "n01768244", "n03085013",
              "n03255030", "n02236044", "n01560419", "n03014705", "n01753488", "n03902125", "n02106166", "n02906734",
              "n04517823", "n02815834", "n03733131", "n04604644", "n02971356", "n02110063", "n02396427", "n02017213",
              "n01592084", "n04039381", "n03417042", "n02749479", "n02129604", "n01687978", "n02917067", "n03179701",
              "n04380533", "n02091467", "n02786058", "n01829413", "n04254120", "n03920288", "n04152593", "n01795545",
              "n01694178", "n02231487", "n03445924", "n03961711", "n02113624", "n04548280", "n02317335", "n02107312",
              "n01729322", "n03691459", "n01693334", "n02783161", "n04458633", "n04523525", "n07760859", "n07583066",
              "n03384352", "n04589890", "n02514041", "n02093256"]

# Web-site is asking to cite paper from 2015.
# http://www.image-net.org/challenges/LSVRC/2012/index#cite
_CITATION = """\
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
"""

_LABELS_FNAME = "image_classification/imagenet2012_labels.txt"

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = "image_classification/imagenet2012_validation_labels.txt"

# From https://github.com/cytsai/ilsvrc-cmyk-image-list
CMYK_IMAGES = [
    "n01739381_1309.JPEG",
    "n02077923_14822.JPEG",
    "n02447366_23489.JPEG",
    "n02492035_15739.JPEG",
    "n02747177_10752.JPEG",
    "n03018349_4028.JPEG",
    "n03062245_4620.JPEG",
    "n03347037_9675.JPEG",
    "n03467068_12171.JPEG",
    "n03529860_11437.JPEG",
    "n03544143_17228.JPEG",
    "n03633091_5218.JPEG",
    "n03710637_5125.JPEG",
    "n03961711_5286.JPEG",
    "n04033995_2932.JPEG",
    "n04258138_17003.JPEG",
    "n04264628_27969.JPEG",
    "n04336792_7448.JPEG",
    "n04371774_5854.JPEG",
    "n04596742_4225.JPEG",
    "n07583066_647.JPEG",
    "n13037406_4650.JPEG",
]

PNG_IMAGES = ["n02105855_2933.JPEG"]


class SubImagenet0_2012(tfds.core.GeneratorBasedBuilder):
    """Imagenet 2012, aka ILSVRC 2012 with only 100 classes."""

    VERSION = tfds.core.Version("5.1.0")
    SUPPORTED_VERSIONS = [
        tfds.core.Version("5.0.0"),
    ]
    RELEASE_NOTES = {
        "5.1.0":
            "Added test split.",
        "5.0.0":
            "New split API (https://tensorflow.org/datasets/splits)",
        "4.0.0":
            "(unpublished)",
        "3.0.0":
            """
      Fix colorization on ~12 images (CMYK -> RGB).
      Fix format for consistency (convert the single png image to Jpeg).
      Faster generation reading directly from the archive.
      """,
        "2.0.1":
            "Encoding fix. No changes from user point of view.",
        "2.0.0":
            "Fix validation labels.",
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain two files: ILSVRC2012_img_train.tar and
  ILSVRC2012_img_val.tar.
  You need to register on http://www.image-net.org/download-images in order
  to get the link to download the dataset.
  """

    def _info(self):
        names_file = tfds.core.tfds_path(_LABELS_FNAME)
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(encoding_format="jpeg"),
                "label": tfds.features.ClassLabel(names=label_list),
                "file_name": tfds.features.Text(),  # Eg: "n15075141_54.JPEG"
            }),
            supervised_keys=("image", "label"),
            homepage="http://image-net.org/",
            citation=_CITATION,
        )

    @staticmethod
    def _get_validation_labels(val_path):
        """Returns labels for validation.

    Args:
      val_path: path to TAR file containing validation images. It is used to
        retrieve the name of pictures and associate them to labels.

    Returns:
      dict, mapping from image name (str) to label (str).
    """
        labels_path = tfds.core.tfds_path(_VALIDATION_LABELS_FNAME)
        with tf.io.gfile.GFile(os.fspath(labels_path)) as labels_f:
            # `splitlines` to remove trailing `\r` in Windows
            labels = labels_f.read().strip().splitlines()
        with tf.io.gfile.GFile(val_path, "rb") as tar_f_obj:
            tar = tarfile.open(mode="r:", fileobj=tar_f_obj)
            images = sorted(tar.getnames())
        return dict(zip(images, labels))

    def _split_generators(self, dl_manager):
        train_path = os.path.join(dl_manager.manual_dir, "ILSVRC2012_img_train.tar")
        val_path = os.path.join(dl_manager.manual_dir, "ILSVRC2012_img_val.tar")
        test_path = os.path.join(dl_manager.manual_dir, "ILSVRC2012_img_test.tar")
        splits = []
        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.TRAIN,
            split_path=train_path,
            dl_manager=dl_manager,
        )
        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.VALIDATION,
            split_path=val_path,
            dl_manager=dl_manager,
            validation_labels=self._get_validation_labels(val_path),
        )
        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.TEST,
            split_path=test_path,
            dl_manager=dl_manager,
            labels_exist=False,
        )
        if not splits:
            raise AssertionError(
                "ImageNet requires manual download of the data. Please download "
                "the data and place them into:\n"
                f" * train: {train_path}\n"
                f" * test: {test_path}\n"
                f" * validation: {val_path}\n"
                "At least one of the split should be available.")
        return splits

    def _fix_image(self, image_fname, image):
        """Fix image color system and format starting from v 3.0.0."""
        if self.version < "3.0.0":
            return image
        if image_fname in CMYK_IMAGES:
            image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
        elif image_fname in PNG_IMAGES:
            image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))
        return image

    def _generate_examples(self,
                           archive,
                           validation_labels=None,
                           labels_exist=True):
        """Yields examples."""
        if not labels_exist:  # Test split
            for key, example in self._generate_examples_test(archive):
                yield key, example
        if validation_labels:  # Validation split
            for key, example in self._generate_examples_validation(
                    archive, validation_labels):
                yield key, example
        # Training split. Main archive contains archives names after a synset noun.
        # Each sub-archive contains pictures associated to that synset.
        for fname, fobj in archive:
            label = fname[:-4]  # fname is something like "n01632458.tar"
            # TODO(b/117643231): in py3, the following lines trigger tarfile module
            # to call `fobj.seekable()`, which Gfile doesn't have. We should find an
            # alternative, as this loads ~150MB in RAM.
            if label in label_list:
                fobj_mem = io.BytesIO(fobj.read())
                for image_fname, image in tfds.download.iter_archive(
                        fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
                    image = self._fix_image(image_fname, image)
                    record = {
                        "file_name": image_fname,
                        "image": image,
                        "label": label,
                    }
                    yield image_fname, record

    def _generate_examples_validation(self, archive, labels):
        for fname, fobj in archive:
            record = {
                "file_name": fname,
                "image": fobj,
                "label": labels[fname],
            }
            if labels[fname] in label_list:
                yield fname, record

    def _generate_examples_test(self, archive):
        for fname, fobj in archive:
            record = {
                "file_name": fname,
                "image": fobj,
                "label": -1,
            }
            yield fname, record


def _add_split_if_exists(split_list, split, split_path, dl_manager, **kwargs):
    """Add split to given list of splits only if the file exists."""
    if not tf.io.gfile.exists(split_path):
        logging.warning(
            "ImageNet 2012 Challenge %s split not found at %s. "
            "Proceeding with data generation anyways but the split will be "
            "missing from the dataset...",
            str(split),
            split_path,
        )
    else:
        split_list.append(
            tfds.core.SplitGenerator(
                name=split,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(split_path),
                    **kwargs
                },
            ), )
