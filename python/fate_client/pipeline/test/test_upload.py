from pipeline.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline()
pipeline.upload(file="/Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/examples/data/breast_hetero_guest.csv",
                head=1,
                partitions=4,
                namespace="experiment",
                name="breast_hetero_guest",
                storage_engine="standalone",
                meta={
                    "label_name": "y",
                    "label_type": "float32",
                    "dtype": "float32"
                })
