import json
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError

import yaml

ESSENTIA_MODELS_SITE = "https://essentia.upf.edu/models"

INPUT_DEFAULTS = {
    "TensorflowPredictMusiCNN": "model/Placeholder",
    "TensorflowPredictVGGish": "model/Placeholder",
    "TensorflowPredict2D": "model/Placeholder",
    "TensorflowPredict": "model/Placeholder",
    "TensorflowPredictEffnetDiscogs": "serving_default_melspectrogram",
    "TensorflowPredictFSDSINet": "x",
    "TensorflowPredictMAEST": "melspectrogram",
    "PitchCREPE": "frames",
    "TempoCNN": "input",
}

OUTPUT_DEFAULTS = {
    "TensorflowPredictMusiCNN": "model/Sigmoid",
    "TensorflowPredictVGGish": "model/Sigmoid",
    "TensorflowPredict2D": "model/Sigmoid",
    "TensorflowPredict": "model/Sigmoid",
    "TensorflowPredictEffnetDiscogs": "PartitionedCall:0",
    "TensorflowPredictFSDSINet": "model/predictions/Sigmoid",
    "TensorflowPredictMAEST": "Identity",
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
        f'audio = MonoLoader(filename="{audio_file}", sampleRate={sample_rate}, resampleQuality=4)()\n'
        f'model = {algo_name}(graphFilename="{graph_filename}"{output_node})\n'
        f"{algo_returns} = model(audio)\n"
    )


def generate_two_steps_algorithm(
    first_graph_filename: str,
    first_algo_name: str,
    first_algo_params: str,
    second_graph_filename: str,
    second_algo_name: str,
    second_algo_parms: str,
    sample_rate: int,
    algo_returns: str,
    audio_file: str,
    output_name: str | None = None,
):
    if second_algo_name == "TensorflowPredict2D":
        return (
            f"from essentia.standard import MonoLoader, {first_algo_name}, {second_algo_name}\n"
            "\n"
            f'audio = MonoLoader(filename="{audio_file}", sampleRate={sample_rate}, resampleQuality=4)()\n'
            f'embedding_model = {first_algo_name}(graphFilename="{first_graph_filename}"{first_algo_params})\n'
            f"embeddings = embedding_model(audio)\n"
            "\n"
            f'model = {second_algo_name}(graphFilename="{second_graph_filename}"{second_algo_parms})\n'
            f"{algo_returns} = model(embeddings)\n"
        )
    elif second_algo_name == "TensorflowPredict":
        assert output_name is not None, (
            "output_name must be specified for TensorflowPredict"
        )
        return (
            "from essentia import Pool\n"
            f"from essentia.standard import MonoLoader, {first_algo_name}, {second_algo_name}\n"
            "\n"
            f'audio = MonoLoader(filename="{audio_file}", sampleRate={sample_rate}, resampleQuality=4)()\n'
            f'embedding_model = {first_algo_name}(graphFilename="{first_graph_filename}"{first_algo_params})\n'
            f"embeddings = embedding_model(audio)\n"
            "\n"
            "pool = Pool()\n"
            'pool.set("embeddings", embeddings)\n'
            "\n"
            f'model = {second_algo_name}(graphFilename="{second_graph_filename}"{second_algo_parms})\n'
            f'{algo_returns} = model(pool)["{output_name}"]\n'
        )
    else:
        raise ValueError(f"Unknown second_algo_name: {second_algo_name}")


def get_output_node_name(metadata: dict, output_purpose: str):
    """Get the output node name for a given output purpose"""
    outputs = metadata["schema"]["outputs"]

    for output in outputs:
        if "output_purpose" not in output:
            continue
        if output["output_purpose"] == output_purpose:
            return output["name"]

    raise ValueError(f"Output node not found for `output_purpose`: {output_purpose}")


def get_kwargs_string(metadata: dict, output_purpose: str, algo_name: str):
    """Get kwargs string for a given algorithm"""

    kwargs_str = ""

    algo_name = metadata["inference"]["algorithm"]
    input = metadata["schema"]["inputs"][0]["name"]

    # Set input related params
    if input != INPUT_DEFAULTS[algo_name]:
        if algo_name == "TensorflowPredict":
            kwargs_str += f', inputs=["{input}"]'
        else:
            kwargs_str += f', input="{input}"'

    # Set output related params
    output_node_name = get_output_node_name(metadata, output_purpose)

    if output_node_name != OUTPUT_DEFAULTS[algo_name]:
        if algo_name == "TensorflowPredict":
            kwargs_str += f', outputs=["{output_node_name}"]'
        else:
            kwargs_str += f', output="{output_node_name}"'

    return kwargs_str


def get_metadata(task_type: str, family_name: str, model: str, metadata_base_dir=False):
    if metadata_base_dir:
        metadata_path = str(
            Path(metadata_base_dir, task_type, family_name, f"{model}.json")
        )
        metadata = read_metadata(metadata_path)
    else:
        metadata_path = "/".join(
            [ESSENTIA_MODELS_SITE, task_type, family_name, f"{model}.json"]
        )
        try:
            metadata = download_metadata(metadata_path)
        except HTTPError:
            print(f"Failed downloading {metadata_path}")
            exit(1)

    return metadata


def process_model(
    task_type: str,
    family_name: str,
    model: str,
    output: str,
    metadata_base_dir: str,
    models_base_dir: str,
    audio_file: str,
    download_models: str,
    script_dir: Path,
):
    print("processing", model)
    metadata = get_metadata(
        task_type,
        family_name,
        model,
        metadata_base_dir=metadata_base_dir,
    )

    # get algorithm name
    algo_name = metadata["inference"]["algorithm"]

    # check if we need a custom output node
    algo_kwargs = get_kwargs_string(metadata, output, algo_name)

    # set algos with custom output
    algo_returns = CUSTOM_ALGO_OUTPUTS.get(algo_name, output)
    graph_filename = Path(f"{model}.pb")
    if models_base_dir:
        graph_filename = Path(models_base_dir, task_type, family_name, graph_filename)

    graph_filename_tgt = script_dir / graph_filename
    if download_models and (not graph_filename_tgt.exists()):
        assert not models_base_dir, (
            "downloading the models is incompatible with specifying `models_base_dir`"
        )
        try:
            script_dir.mkdir(parents=True, exist_ok=True)
            urlretrieve(metadata["link"], graph_filename_tgt)
        except HTTPError:
            print(f"Failed downloading {metadata['link']}")
            exit(1)

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
            metadata_base_dir=metadata_base_dir,
        )
        embedding_graph_filename = Path(embedding_model_name + ".pb")

        if models_base_dir:
            embedding_graph_filename = Path(
                models_base_dir,
                embedding_task_type,
                embedding_family_name,
                embedding_graph_filename,
            )

        embedding_graph_filename_tgt = script_dir / embedding_graph_filename
        if download_models and not embedding_graph_filename_tgt.exists():
            try:
                urlretrieve(embedding_metadata["link"], embedding_graph_filename_tgt)
            except HTTPError:
                print(f"Failed downloading {metadata['link']}")
                exit(1)

        embedding_algo_kwargs = get_kwargs_string(
            embedding_metadata,
            "embeddings",
            embedding_algo_name,
        )
        output_node_name = get_output_node_name(metadata, output)

        # Exceptions:
        # - MAEST-based genre discogs models use the 12th layer instead of the 7th
        if "Genre Discogs" in metadata["name"]:
            embedding_algo_kwargs = embedding_algo_kwargs.replace("7", "12")

        script = generate_two_steps_algorithm(
            embedding_graph_filename,
            embedding_algo_name,
            embedding_algo_kwargs,
            graph_filename,
            algo_name,
            algo_kwargs,
            sample_rate,
            algo_returns,
            audio_file,
            output_name=output_node_name,
        )
    else:
        script = generate_single_step_algorithm(
            graph_filename,
            algo_name,
            sample_rate,
            algo_kwargs,
            algo_returns,
            audio_file,
        )

    return script


def generate_example_scripts(
    output_dir: Path,
    models_base_dir: str = "",
    metadata_base_dir: str = "",
    audio_file: str = "audio.wav",
    force: bool = False,
    format_with_black: bool = False,
    download_models: bool = False,
):
    models_file = Path(__file__).parent / "models.yaml"
    with open(models_file, "r") as models_f:
        models_data = yaml.load(models_f, Loader=yaml.FullLoader)

    scripts = []

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
                        metadata_base_dir=metadata_base_dir,
                        models_base_dir=models_base_dir,
                        audio_file=audio_file,
                        download_models=download_models,
                        script_dir=file.parent,
                    )

                    if not file.exists():
                        file.parent.mkdir(exist_ok=True, parents=True)

                    with open(file, "w") as script_file:
                        script_file.write(script)

                    scripts.append(str(file))
                    print(f"generated script {file}")

    return scripts


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
        "--models-base-dir",
        default="",
        help="whether to set model paths relative to a given directory",
    )
    parser.add_argument(
        "--audio-file",
        default="audio.wav",
        help="the audio file to use in the audio examples",
    )
    parser.add_argument(
        "--metadata-base-dir",
        type=str,
        default="",
        help="if not empty, .json files are read from this directory",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="whether to download models next to the script location",
    )
    args = parser.parse_args()

    generate_example_scripts(
        Path(args.output_dir).resolve(),
        models_base_dir=args.models_base_dir,
        metadata_base_dir=args.metadata_base_dir,
        audio_file=args.audio_file,
        force=args.force,
        download_models=args.download_models,
    )
