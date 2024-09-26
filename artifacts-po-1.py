from typing import TypeVar, Annotated, Tuple

from flytekit import Artifact
from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.artifact import ArtifactGroup, REPORT, create_artifact
from flytekitplugins.domino.helpers import DominoJobTask, DominoJobConfig

# key pieces of data to collect

# artifact groups are identified uniquely within an execution by composite key
# * name
# * type

# artifact files within each group are identified uniquely by filename and graph node
# * filename
# * graph node
# * artifact group (foreign key)
# NOTE: there is an edge case where different outputs in the same node could be annotated identically - we'll require validation to prevent this

# the problem is the way that Flyte stores metadata and how the existing development experience works
# it makes it cumbersome to place artifacts into specific groups b/c of how the Python types are defined

# we need to change the DX because:
# it's error prone
# requires specifying the group values as partitions again and again
# requires more code than should be necessary, including predefining Artifacts by name instead of inside the Annotation
# the name of the artifact should be able to automatically set the extension (used by frontend for file previews)

# also note its worth investigating behavior dynamic partitions - i.e. ReportArtifact.create_from()

# upstream code here shows some examples
# https://github.com/flyteorg/flytekit/blob/master/flytekit/core/artifact.py#L371
# https://github.com/flyteorg/flytekit/blob/master/tests/flytekit/unit/core/test_artifacts.py

ReportGroup1 = ArtifactGroup(name="report_foo", kind=REPORT)
ReportGroup2 = ArtifactGroup(name="report_bar", kind=REPORT)

# to use partition_keys (necessary for Domino), we have to define this type up front
ReportArtifact1 = Artifact(name="report1.pdf", partition_keys=["group", "type"])(group="report_foo", type="report")
ReportArtifact2 = create_artifact(name="report2.pdf", group=ReportGroup1)
ReportArtifact3 = create_artifact(name="report3.pdf", group=ReportGroup2)

# ideally, a group is defined like this
# ReportGroup = Group(name="my custom report", type=Report)


@workflow
def artifact_meta(data_path: str) -> Tuple[
    Annotated[FlyteFile, ReportArtifact1],
    Annotated[FlyteFile, ReportArtifact2],
    Annotated[FlyteFile, ReportArtifact3],

    # ideally the definition looks more like this:
    # Annotated[FlyteFile, Artifact(name="report.pdf", Group=ReportGroup)],
    # this could be further simplified in the programming model if we know that these artifacts are only a single file like
    # ArtifactFile(name="report.pdf", Group=ReportGroup)

    # normal workflow output with no annotations
    FlyteFile
]:
    """py
    pyflyte run --remote artifacts-po-1.py artifact_meta --data_path /mnt/code/data/data.csv
    """

    data_prep_results = DominoJobTask(
        name="Prepare data",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/prep-data.py",
        ),
        inputs={
            "data_path": str
        },
        outputs={
            # NOTE: Flyte normally suppports this -- but notice there are no partitions, which make them useless to Domino
            # this output is consumed by a subsequent task but also marked as an artifact
            "processed_data": Annotated[FlyteFile[TypeVar("csv")], Artifact(name="processed.csv")],
            # no downstream consumers -- simply an artifact output from an intermediate node in the graph
            "processed_data2": Annotated[FlyteFile[TypeVar("csv")], Artifact(name="processed2.csv")],
        },
        use_latest=True,
    )(data_path=data_path)

    training_results = DominoJobTask(
        name="Train model",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/train-model.py",
        ),
        inputs={
            "processed_data": FlyteFile[TypeVar("csv")],
            "epochs": int,
            "batch_size": int,
        },
        outputs={
            "model": FlyteFile[TypeVar("csv")],
        },
        use_latest=True,
    )(processed_data=data_prep_results.processed_data, epochs=10, batch_size=32)

    # return the result from 2nd node to the workflow annotated in different ways
    model = training_results['model']

    return (
        model,
        model,
        model,
        model,
    )
