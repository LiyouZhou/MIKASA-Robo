#! /usr/bin/env python3

import re

readme_file = "../README.md"

with open(readme_file, "r") as f:
    readme = f.read()

matches = re.findall(
    r"\[Download .* dataset\]\((https://.*\.zip)\)", readme)


for match in matches:
    print("url:", str(match))
    print("name:", str(match).split("/")[-1].split(".")[0])
