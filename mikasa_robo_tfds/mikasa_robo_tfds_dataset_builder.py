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

        assert "prompt" in kwargs, "Prompt must be provided in kwargs"
        self.prompt = kwargs.pop("prompt", "No Prompt")

        super(MikasaRoboTfdsConfig, self).__init__(
            version=tfds.core.Version("1.0.1"),
            **kwargs,
        )


def RememberColorBaselinePromptBuilder(task_name, data):
    baseline = "baseline" in task_name
    # find the color of the cube that need to be touched
    if task_name == "RememberColor3-v0_baseline":
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
    elif task_name == "RememberColor9-v0_baseline":
        clean_data = remove(data["rgb"][1][35:60, 60:85, 3:])
        all_pixels_in_image = []
        for i in range(len(clean_data)):
            for j in range(len(clean_data[i])):
                hsv = rgb2hsv(clean_data[i][j][:3])
                all_pixels_in_image.append(hsv)
        all_color_pixels_in_image = [x for x in all_pixels_in_image if x[1] > 0.9]

        color_hue = sum([x[0] for x in all_color_pixels_in_image]) / len(
            all_color_pixels_in_image
        )
        color_name_index = np.argmin([abs(x - color_hue) for x in HUE_NAMES.keys()])
        colour = HUE_NAMES[list(HUE_NAMES.keys())[color_name_index]]
    else:
        raise ValueError("Unknown dataset")

    return f"touch the {colour} cube"


# fmt: off
PROMPT_TEMPLATE_SHELL_GAME_TOUCH = "Memorize the position of the ball, then touch the cup with ball."
PROMPT_TEMPLATE_SHELL_GAME_PUSH = "Memorize the position of the ball, then push the cup with ball."
PROMPT_TEMPLATE_SHELL_GAME_PICK = "Memorize the position of the ball, then pick up the cup with ball."
PROMPT_TEMPLATE_INTERCEPT = "Intercept the rolling ball and guide it towards the target."
PROMPT_TEMPLATE_INTERCEPT_GRAB = "Intercept the rolling ball. Then catch the ball with the gripper and lift it up."
PROMPT_TEMPLATE_ROTATE_LENIENT = "Memorize the initial position of the peg and rotate it back to its initial position."
PROMPT_TEMPLATE_ROTATE_STRICT = "Memorize the initial position of the peg and rotate it back to its initial position without shifting its center."
PROMPT_TEMPLATE_TAKE_IT_BACK = "Memorize the initial position of the cube, move it to the target region, and then return it to its initial position."
PROMPT_TEMPLATE_REMEMBER_COLOR = "Memorize the the colors of the cube shown on the table, and then touch the same coloured cube out of all the cubes."
PROMPT_TEMPLATE_REMEMBER_SHAPE = "Memorize the the shapes of the block shown on the table, and then touch the same shaped blocks."
PROMPT_TEMPLATE_REMEMBER_SHAPE_AND_COLOR = "Memorize the shape and color of the blocks shown, and touch the blocks with the same shape and color."
PROMPT_TEMPLATE_BUNCH_OF_COLORS = "Remember colors of the blocks shown at the begining, touch the same colored blocks in any order."
PROMPT_TEMPLATE_SEQ_OF_COLORS = "Remember the colors of the set of cubes shown sequentially and then touch them in any order."
PROMPT_TEMPLATE_CHAIN_OF_COLORS = "Remember the colors of the set of cubes shown sequentially and then select them in the same order as shown."
# fmt: on

TASK_PROMPTS = {
    "ShellGameTouch-v0": PROMPT_TEMPLATE_SHELL_GAME_TOUCH,
    "ShellGamePush-v0": PROMPT_TEMPLATE_SHELL_GAME_PUSH,
    "ShellGamePick-v0": PROMPT_TEMPLATE_SHELL_GAME_PICK,
    "InterceptSlow-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptMedium-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptFast-v0": PROMPT_TEMPLATE_INTERCEPT,
    "InterceptGrabSlow-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "InterceptGrabMedium-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "InterceptGrabFast-v0": PROMPT_TEMPLATE_INTERCEPT_GRAB,
    "RotateLenientPos-v0": PROMPT_TEMPLATE_ROTATE_LENIENT,
    "RotateLenientPosNeg-v0": PROMPT_TEMPLATE_ROTATE_LENIENT,
    "RotateStrictPos-v0": PROMPT_TEMPLATE_ROTATE_STRICT,
    "RotateStrictPosNeg-v0": PROMPT_TEMPLATE_ROTATE_STRICT,
    "TakeItBack-v0": PROMPT_TEMPLATE_TAKE_IT_BACK,
    "RememberColor3-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberColor5-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberColor9-v0": PROMPT_TEMPLATE_REMEMBER_COLOR,
    "RememberShape3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShape5-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShape9-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor3x2-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor3x3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "RememberShapeAndColor5x3-v0": PROMPT_TEMPLATE_REMEMBER_SHAPE,
    "BunchOfColors3-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "BunchOfColors5-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "BunchOfColors7-v0": PROMPT_TEMPLATE_BUNCH_OF_COLORS,
    "SeqOfColors3-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "SeqOfColors5-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "SeqOfColors7-v0": PROMPT_TEMPLATE_SEQ_OF_COLORS,
    "ChainOfColors3-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
    "ChainOfColors5-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
    "ChainOfColors7-v0": PROMPT_TEMPLATE_CHAIN_OF_COLORS,
}


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
            prompt="default prompt",
        ),
        MikasaRoboTfdsConfig(
            name="RememberColor9-v0_baseline",
            description="Default configuration for mikasa_robo_tfds dataset.",
            data_dir="../data/RememberColor9-v0/",
            url="https://huggingface.co/datasets/avanturist/mikasa-robo/resolve/main/RememberColor9-v0.zip",
            prompt=RememberColorBaselinePromptBuilder,
        ),
        MikasaRoboTfdsConfig(
            name="RememberColor3-v0_baseline",
            description="Default configuration for mikasa_robo_tfds dataset.",
            data_dir="../data/RememberColor3-v0/",
            url="https://huggingface.co/datasets/avanturist/mikasa-robo/resolve/main/RememberColor3-v0.zip",
            prompt=RememberColorBaselinePromptBuilder,
        ),
    ]

    def __init__(self, **kwargs):
        readme_file = "../README.md"

        with open(readme_file, "r") as f:
            readme = f.read()

        matches = re.findall(r"\[Download .* dataset\]\((https://.*\.zip)\)", readme)

        for match in matches:
            url = str(match)
            name = str(match).split("/")[-1].split(".")[0]

            self.BUILDER_CONFIGS.append(
                MikasaRoboTfdsConfig(
                    name=name,
                    description=f"mikasa_robo_tfds dataset task {name}",
                    data_dir=f"../data/{name}",
                    url=url,
                    prompt=TASK_PROMPTS[name] if name in TASK_PROMPTS else "No Prompt",
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
                                    "image2": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="secondary image",
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
                            "task_name": tfds.features.Text(
                                doc="Name of the task, e.g., 'ShellGameTouch-v0'."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        archive_path = dl_manager.download_and_extract(self.builder_config.url)

        data_dir = Path(archive_path) / self.builder_config.name.strip("_baseline")
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
            """Parses a single episode file."""

            # load the episode data
            data = np.load(
                episode_path, allow_pickle=True
            )  # this is a list of dicts in our case

            # create the language instruction for this episode
            if callable(self.builder_config.prompt):
                language_instruction = self.builder_config.prompt(
                    self.builder_config.name, data
                )
            elif isinstance(self.builder_config.prompt, str):
                language_instruction = self.builder_config.prompt
            else:
                raise ValueError(
                    "Prompt must be a string or a callable that returns a string."
                )

            episode = []
            for i in range(len(data["rgb"]) - 1):
                # https://github.com/CognitiveAISystems/MIKASA-Robo/issues/3#issuecomment-2772414275
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
                        data["action"][i][
                            -1
                        ],  # gripper always controlled by absolute position
                    ],
                    dtype=np.float32,
                )
                state = data["joints"][i].astype(np.float32)
                is_terminal = bool(data["done"][i] or data["success"][i])
                image = data["rgb"][i]

                episode.append(
                    {
                        "is_first": i == 0,
                        "is_last": i == len(data) - 1,
                        "observation": {
                            "image": image[:, :, :3],  # only take top view image
                            "image2": image[:, :, 3:],  # secondary image
                            "state": state,
                        },
                        "action": action,
                        "is_terminal": is_terminal,
                        "language_instruction": language_instruction,
                        "step": i,
                    }
                )

                # stop if terminal state is reached
                if is_terminal:
                    episode[-1]["is_last"] = True
                    break

            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(episode_path),
                    "task_name": self.builder_config.name,
                },
            }

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
