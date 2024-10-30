from typing import List, Dict, NamedTuple

from flytekit import workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.artifact import Artifact, DATA, REPORT
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask, GitRef

ReportArtifact = Artifact(name="My Report", type=REPORT)
DataArtifact = Artifact(name="My Data", type=DATA)

final_outputs = NamedTuple(
    "final_outputs",
    csv=DataArtifact.File(name="data.csv"),
    notebook=DataArtifact.File(name="notebook.ipynb"),
    report=ReportArtifact.File(name="generated.pdf")
)


# pyflyte run --remote generate_artifacts.py generate_artifacts
@workflow
def generate_artifacts() -> final_outputs:
    sce_types = DominoJobTask(
        name="Generate SCE Types",
        domino_job_config=DominoJobConfig(
            MainRepoGitRef=GitRef(Type="head"),
            Command="python /mnt/code/scripts/generate-sce-types.py",
        ),
        inputs={
            'sdtm_data_path': str,
        },
        outputs={
            'pdf': ReportArtifact.File(name="generated.pdf"),
            'sas7bdat': DataArtifact.File(name="data.sas7bdat"),
        },
        use_latest=True
    )

    sce_returns = sce_types(sdtm_data_path="/mnt/code/artifacts")

    ml_types = DominoJobTask(
        name="Generate ML Types",
        domino_job_config=DominoJobConfig(
            MainRepoGitRef=GitRef(Type="head"),
            Command="python /mnt/code/scripts/generate-ml-types.py",
        ),
        inputs={
            'batch_size': int,
            'learning_rate': float,
            'do_eval': bool,
            'list': List[int],
            'dict': Dict[str, int]
        },
        outputs={
            'csv': FlyteFile["csv"],
            'json': DataArtifact.File(name="data.json"),
            'png': ReportArtifact.File(name="report.png"),
            'jpeg': ReportArtifact.File(name="report.jpeg"),
            'notebook': FlyteFile["ipynb"],
            'mlflow_model': FlyteDirectory
        },
        use_latest=True
    )

    ml_returns = ml_types(
        batch_size=32,
        learning_rate=0.001,
        do_eval=True,
        list=[1, 2, 3, 4, 5],
        dict={"param1": 1, "param2": 2, "param3": 3}
    )

    # TODO: why is this ml_returns[0] and not ml_returns["csv"] ? Something looks broken
    return final_outputs(
        csv=ml_returns.csv,
        notebook=ml_returns.notebook,
        report=sce_returns.pdf,
    )
