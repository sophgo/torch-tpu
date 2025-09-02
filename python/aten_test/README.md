## A tool to extract all op in models by profile and run tpudnn batch testing comparing with cpu results
model_factory provides a series of well-defined model for test

however, complex models like yolo may need to put the profile code in fwd-bwd to trace op

and we provided a yolo_ops.json for a straight test by `python generate_test.py --json test_ops.json`

Example:
```
python export.py --model lenet --batch_size 64
```
Two JSON files will be generated:
- trace.json includes all information
- test_ops.json will be used by batch testing

```
python generate_test.py --json test_ops.json
```