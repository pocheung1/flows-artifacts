from typing import Annotated, Tuple

from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.domino.artifact import ArtifactGroup, REPORT, annotated_artifact_file, artifact_specification
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
# it's error-prone
# requires specifying the group values as partitions again and again
# requires more code than should be necessary, including predefining Artifacts by name instead of inside the Annotation
# the name of the artifact should be able to automatically set the extension (used by frontend for file previews)

# also note its worth investigating behavior dynamic partitions - i.e. ReportArtifact.create_from()

# upstream code here shows some examples
# https://github.com/flyteorg/flytekit/blob/master/flytekit/core/artifact.py#L371
# https://github.com/flyteorg/flytekit/blob/master/tests/flytekit/unit/core/test_artifacts.py

# define artifact groups
reports_foo = ArtifactGroup(name="reports_foo", type=REPORT)
reports_bar = ArtifactGroup(name="reports_bar", type=REPORT)


@workflow
def artifact_meta(data_path: str) -> Tuple[
    # annotated workflow output with group partitions
    annotated_artifact_file(name="report1.csv", group=reports_foo),
    # override file extension with file type
    annotated_artifact_file(name="report2.pdf", group=reports_foo, file_type="csv"),
    # override missing file extension with file type
    annotated_artifact_file(name="report3", group=reports_bar, file_type="csv"),
    # normal workflow output with no annotations
    FlyteFile["csv"],
]:
    """py
    pyflyte run --remote artifacts.py artifact_meta --data_path /mnt/code/data/data.csv
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
            # this output is consumed by a subsequent task but also marked as an artifact with partitions
            "processed_data": annotated_artifact_file(name="processed.csv", group=reports_foo),
            # no downstream consumers -- simply an artifact output from an intermediate node in the graph
            "processed_data2": Annotated[FlyteFile["csv"], artifact_specification(name="processed2.csv", group=reports_bar)],
        },
        use_latest=True,
    )(data_path=data_path)

    training_results = DominoJobTask(
        name="Train model",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/train-model.py",
        ),
        inputs={
            "processed_data": FlyteFile["csv"],
            "epochs": int,
            "batch_size": int,
        },
        outputs={
            "model": FlyteFile["csv"],
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
