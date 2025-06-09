"""mikasa_robo_tfds dataset."""

import tensorflow_datasets as tfds
import numpy as np
import glob
from pathlib import Path
import random
from scipy.spatial.transform import Rotation as R
from rembg import remove
from skimage.color import rgb2hsv
import re

HUE_NAMES = {
    0.005: "red",
    0.015: "maroon",
    0.123: "orange",
    0.155: "yellow",
    0.330: "green",
    0.477: "teal",
    0.500: "cyan",
    0.574: "blue",
    0.746: "purple",
}


class MikasaRoboTfdsConfig(tfds.core.BuilderConfig):
    """Configuration for MikasaRoboTfds."""

    def __init__(self, **kwargs):
        """BuilderConfig for MikasaRoboTfds.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.data_dir = kwargs.pop("data_dir", "../data")
        self.url = kwargs.pop("url", "../data")
        super(MikasaRoboTfdsConfig, self).__init__(
            version=tfds.core.Version("1.0.0"),
            **kwargs,
        )


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mikasa_robo_tfds dataset."""

    # VERSION = tfds.core.Version("1.0.0")
    # RELEASE_NOTES = {
    #     "1.0.0": "Initial release.",
    # }

    BUILDER_CONFIGS = [
        MikasaRoboTfdsConfig(
            name="default",
            description="Default configuration for mikasa_robo_tfds dataset.",
            data_dir="../data/",
            url="",
        ),
        MikasaRoboTfdsConfig(
            name="RememberColor9-v0_baseline",
            description="Default configuration for mikasa_robo_tfds dataset.",
            data_dir="../data/RememberColor9-v0/",
            url="",
        ),
        MikasaRoboTfdsConfig(
            name="RememberColor3-v0_baseline",
            description="Default configuration for mikasa_robo_tfds dataset.",
            data_dir="../data/RememberColor3-v0/",
            url="",
        ),
    ]

    def __init__(self, **kwargs):
        readme_file = "../README.md"

        with open(readme_file, "r") as f:
            readme = f.read()

        matches = re.findall(
            r"\[Download .* dataset\]\((https://.*\.zip)\)", readme)

        for match in matches:
            url = str(match)
            name = str(match).split("/")[-1].split(".")[0]

            self.BUILDER_CONFIGS.append(
                MikasaRoboTfdsConfig(
                    name=name,
                    description=f"mikasa_robo_tfds dataset task {name}",
                    data_dir=f"../data/{name}",
                    url=url,
                )
            )

        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="camera image.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(25,),
                                        dtype=np.float32,
                                        doc="absolute x y z yaw (world coordinate)",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="dx dy dz dyaw",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Natural language instruction for the task."
                            ),
                            "step": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Step number in the episode.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        archive_path = dl_manager.download_and_extract(
            self.builder_config.url
        )

        data_dir = Path(archive_path) / self.builder_config.name
        print(f"Using data from {data_dir}")
        episodes = data_dir.glob("*.npz")

        # split the episodes into train and val
        episodes = list(episodes)
        random.seed(42)
        train_set_len = int(0.8 * len(episodes))
        train_set = set(random.sample(episodes, train_set_len))
        val_set = set(episodes) - train_set

        return {
            "train": self._generate_examples(
                paths=list(train_set),
            ),
            "val": self._generate_examples(
                paths=list(val_set),
            ),
        }

    def _generate_examples(self, paths):
        """Yields examples."""

        def _parse_episode(episode_path):
            # episode_path = tf.io.gfile.GFile(episode_path, mode='r')
            # load raw data --> this should change for your dataset
            baseline = "baseline" in self.builder_config.name

            data = np.load(
                episode_path, allow_pickle=True
            )  # this is a list of dicts in our case

            if baseline:
                # find the color of the cube that need to be touched
                if "3" in self.builder_config.name:
                    rgb = data["rgb"][1][..., 3:][58, 66]
                    hue = rgb2hsv(np.array(rgb))[0]
                    colour = ""
                    if hue < 0.1:
                        colour = "red"
                    elif hue > 0.3 and hue < 0.4:
                        colour = "green"
                    elif hue > 0.6 and hue < 0.7:
                        colour = "blue"
                    else:
                        raise ValueError("Unknown color")
                elif "9" in self.builder_config.name:
                    clean_data = remove(data["rgb"][1][35:60, 60:85, 3:])
                    all_pixels_in_image = []
                    for i in range(len(clean_data)):
                        for j in range(len(clean_data[i])):
                            hsv = rgb2hsv(clean_data[i][j][:3])
                            all_pixels_in_image.append(hsv)
                    all_color_pixels_in_image = [
                        x for x in all_pixels_in_image if x[1] > 0.9
                    ]

                    color_hue = sum(
                        [x[0] for x in all_color_pixels_in_image]
                    ) / len(all_color_pixels_in_image)
                    color_name_index = np.argmin(
                        [abs(x - color_hue) for x in HUE_NAMES.keys()]
                    )
                    colour = HUE_NAMES[list(HUE_NAMES.keys())[color_name_index]]
                else:
                    raise ValueError("Unknown dataset")

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data["rgb"]) - 1):
                joints = data["joints"][i].astype(np.float32)
                next_joints = data["joints"][i + 1].astype(np.float32)
                ee_pos = joints[:3]
                ee_rot = R.from_quat(joints[3:7])
                next_ee_pos = next_joints[:3]
                next_ee_rot = R.from_quat(next_joints[3:7])
                delta_pos = next_ee_pos - ee_pos
                # calculate the euler angles between the two quaternions
                delta_rot = next_ee_rot * ee_rot.inv()

                action = np.array(
                    [
                        delta_pos[0],
                        delta_pos[1],
                        delta_pos[2],
                        delta_rot.as_euler("xyz")[0],
                        delta_rot.as_euler("xyz")[1],
                        delta_rot.as_euler("xyz")[2],
                        data["action"][i][-1],
                    ],
                    dtype=np.float32,
                )
                state = data["joints"][i].astype(np.float32)
                is_terminal = bool(data["done"][i])
                image = data["rgb"][i - 1]

                episode.append(
                    {
                        "is_first": i == 0,
                        "is_last": i == len(data) - 1,
                        "observation": {
                            "image": image[:, :, :3],  # only take top view image
                            "state": state,
                        },
                        "action": action,
                        "is_terminal": is_terminal,
                        "language_instruction": (
                            f"touch the {colour} cube"
                            if baseline
                            else "Memorize the the colors of the cube shown on the table, and then touch the same coloured cube out of all the cubes."
                        ),
                        "step": i,
                    }
                )
                action_prev = action
            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {"file_path": str(episode_path)},
            }
            # if you want to skip an example for whatever reason, simply return None
            return str(episode_path), sample

        # for smallish datasets, use single-thread parsing
        for episode_path in paths:
            yield _parse_episode(episode_path)


if __name__ == "__main__":
    # test the dataset
    builder = Builder()
    ds = builder.as_dataset(split="train")
    for example in ds:
        print(example)
