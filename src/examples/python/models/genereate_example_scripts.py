import json
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen
from subprocess import run

import yaml

ESSENTIA_MODELS_SITE = "https://essentia.upf.edu/models"
ESSENTIA_MODELS_DIR = "/home/pablo/reps/essentia/src/examples/python/models/new_jsons"

INPUT_DEFAULTS = {
    "TensorflowPredictMusiCNN": "model/Placeholder",
    "TensorflowPredictVGGish": "model/Placeholder",
    "TensorflowPredict2D": "model/Placeholder",
    "TensorflowPredictEffnetDiscogs": "serving_default_melspectrogram",
    "TensorflowPredictFSDSINet": "x",
    "PitchCREPE": "frames",
    "TempoCNN": "input",
}

OUTPUT_DEFAULTS = {
    "TensorflowPredictMusiCNN": "model/Sigmoid",
    "TensorflowPredictVGGish": "model/Sigmoid",
    "TensorflowPredict2D": "model/Sigmoid",
    "TensorflowPredictEffnetDiscogs": "PartitionedCall",
    "TensorflowPredictFSDSINet": "model/predictions/Sigmoid",
    "PitchCREPE": "model/classifier/Sigmoid",
    "TempoCNN": "output",
}

CUSTOM_ALGO_OUTPUTS = {
    "PitchCREPE": "time, frequency, confidence, activations",
    "TempoCNN": "global_tempo, local_tempo, local_tempo_probabilities",
}


def download_metadata(url: str):
    output = urlopen(url).read()
    text = output.decode("utf-8")
    return json.loads(text)


def read_metadata(path: str):
    return json.load(open(path, "r"))


def generate_single_step_algorithm(
    graph_filename: str,
    algo_name: str,
    sample_rate: int,
    output_node: str,
    algo_returns: str,
    audio_file: str,
):
    return (
        f"from essentia.standard import MonoLoader, {algo_name}\n"
        f"\n"
        f'audio = MonoLoader(filename="{audio_file}", sampleRate={sample_rate})()\n'
        f'model = {algo_name}(graphFilename="{graph_filename}"{output_node})\n'
        f"{algo_returns} = model(audio)\n"
    )


def generate_two_steps_algorithm(
    first_graph_filename: str,
    first_algo_name: str,
    first_output_node: str,
    second_graph_filename: str,
    second_algo_name: str,
    second_output_node: str,
    sample_rate: int,
    algo_returns: str,
    audio_file: str,
):
    return (
        f"from essentia.standard import MonoLoader, {first_algo_name}, {second_algo_name}\n"
        "\n"
        f'audio = MonoLoader(filename="{audio_file}", sampleRate={sample_rate})()\n'
        f'embedding_model = {first_algo_name}(graphFilename="{first_graph_filename}"{first_output_node})\n'
        f"embeddings = embedding_model(audio)\n"
        "\n"
        f'model = {second_algo_name}(graphFilename="{second_graph_filename}"{second_output_node})\n'
        f"{algo_returns} = model(embeddings)\n"
    )


def get_additional_parameters(metadata: dict, output: str, algo_name: str):
    additional_parameters = ""

    input = metadata["schema"]["inputs"][0]["name"]
    if input != INPUT_DEFAULTS[algo_name]:
        additional_parameters = f', input="{input}"'

    outputs = metadata["schema"]["outputs"]
    for model_output in outputs:
        if (
            model_output["output_purpose"] == output
            and model_output["name"] != OUTPUT_DEFAULTS[algo_name]
        ):
            additional_parameters += f', output="{model_output["name"]}"'
    return additional_parameters


def get_metadata(task_type: str, family_name: str, model: str, local_jsons=False):
    if local_jsons:
        metadata_path = (
            ESSENTIA_MODELS_DIR
            + "/"
            + task_type
            + "/"
            + family_name
            + "/"
            + f"{model}.json"
        )
        metadata = read_metadata(metadata_path)
    else:
        metadata_path = str(
            Path(ESSENTIA_MODELS_SITE, task_type, family_name, f"{model}.json")
        )
        metadata = download_metadata(metadata_path)

    return metadata


def process_model(
    task_type: str,
    family_name: str,
    model: str,
    output: str,
    local_jsons: bool,
    models_from_folder: str,
    audio_file: str,
):
    metadata = get_metadata(task_type, family_name, model, local_jsons=local_jsons)

    # get algorithm name
    algo_name = metadata["inference"]["algorithm"]

    # check if we need a custom output node
    additional_parameters = get_additional_parameters(metadata, output, algo_name)

    # set algos with custom output
    algo_returns = CUSTOM_ALGO_OUTPUTS.get(algo_name, output)
    graph_filename = f"{model}.pb"
    if models_from_folder:
        graph_filename = Path(
            models_from_folder, task_type, family_name, graph_filename
        )
    sample_rate = metadata["inference"]["sample_rate"]

    if task_type == "classification-heads":
        embedding_model_name = metadata["inference"]["embedding_model"]["model_name"]
        embedding_algo_name = metadata["inference"]["embedding_model"]["algorithm"]
        metadata_link = metadata["inference"]["embedding_model"]["link"]
        embedding_task_type = Path(metadata_link).parent.parent.stem
        embedding_family_name = Path(metadata_link).parent.stem
        embedding_metadata = get_metadata(
            embedding_task_type,
            embedding_family_name,
            embedding_model_name,
            local_jsons=local_jsons,
        )
        embedding_graph_filename = embedding_model_name + ".pb"
        if models_from_folder:
            embedding_graph_filename = Path(
                models_from_folder,
                embedding_task_type,
                embedding_family_name,
                embedding_graph_filename,
            )
        embedding_additional_parameters = get_additional_parameters(
            embedding_metadata, "embeddings", embedding_algo_name
        )

        script = generate_two_steps_algorithm(
            embedding_graph_filename,
            embedding_algo_name,
            embedding_additional_parameters,
            graph_filename,
            algo_name,
            additional_parameters,
            sample_rate,
            algo_returns,
            audio_file,
        )
    else:
        script = generate_single_step_algorithm(
            graph_filename,
            algo_name,
            sample_rate,
            additional_parameters,
            algo_returns,
            audio_file,
        )

    return script


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="whether to recompute existing scripts",
    )
    parser.add_argument(
        "--output-dir", "-o", default="scripts", help="path to store the output scripts"
    )
    parser.add_argument(
        "--models-from-folder",
        default="",
        help="whether to se a relative path for the models",
    )
    parser.add_argument(
        "--audio-file",
        default="audio.wav",
        help="the audio file to use in the audio examples",
    )
    args = parser.parse_args()
    force = args.force
    output_dir = Path(args.output_dir)
    models_from_folder = args.models_from_folder
    audio_file = args.audio_file

    models_file = Path(__file__).parent / "models.yaml"
    with open(models_file, "r") as models_f:
        models_data = yaml.load(models_f, Loader=yaml.FullLoader)

    for task_type, task_data in models_data.items():
        for family_name, family_data in task_data.items():
            for model in family_data["models"]:
                if "openl3" in model:
                    continue
                for output in family_data["outputs"]:
                    file = output_dir / task_type / family_name / f"{model}_{output}.py"
                    if file.exists() and not force:
                        continue

                    script = process_model(
                        task_type,
                        family_name,
                        model,
                        output,
                        local_jsons=True,
                        models_from_folder=models_from_folder,
                        audio_file=audio_file,
                    )

                    if not file.exists():
                        file.parent.mkdir(exist_ok=True, parents=True)

                    with open(file, "w") as script_file:
                        script_file.write(script)

                    print(f"generated script {file}")

    # format scripts with black if available
    try:
        run(["black", "--line-length", "120", output_dir], check=True)
    except:
        print("Black formatter not available")