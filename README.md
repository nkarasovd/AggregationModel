# Aggregation Model

## Data

### Format

Our code works with data presented in the following format:

- Reports directory, where all reports stores in separate JSON files and have names like id.json, where `id` is a report
  id.
- CSV file with events of addition some report to a certain group.

#### Report format
Every report in the reports directory is stored in a separate file in the following format:
```json
{
  "id": 566,
  "timestamp": 1234567891234,
  "class": "java.lang.Exception",
  "frames": [
    [
      "java.util.ArrayList.get",
      "com.company.Class1.method1",
      "com.company.Class2.method2",
      "com.company.Class1.method2"
    ]
  ]
}
```

#### Events format
Every event in events file means that some report with id `rid` was attached to group with id `iid` 
at the time moment `ts`.
Additional column `label` shows the fact, that the event was done automatically or manually.
`label=True` means manual labeling and `label=False` automated attach.

Example of events file:
```
ts,rid,iid,label
906750420,755,1,True
917921167,1106,23,True
922132797,1329,45,False
922133018,1331,31,True
```

### NetBeans

NetBeans dataset has been introduced in "S3M: Siamese Stack (Trace) Similarity Measure" paper and stored on
[Figshare](https://figshare.com/articles/dataset/netbeans_stacktraces_json/14135003) in JSON format.

To convert this JSON file to our format, please use the following converter:

```bash
python src/scripts/state_to_events_converter.py
  --state_path path_to_netbeans.json
  --reports_path dir_path_for_saving_reports
  --events_path path_to_events.csv
```

This script produces the directory with reports and csv file with events.

## Usage

### Similarity model training

Example of running baseline similarity methods on NetBeans dataset:

```bash
python src/similarity/scripts/similarity.py
  --method lerch 
  --data_name netbeans 
  --actions_file path_to_events.csv 
  --reports_path reports_dir 
  --train_start 350 
  --train_longitude 3850 
  --val_start 4200 
  --val_longitude 140 
  --test_start 4340 
  --test_longitude 700
  --model_save_path path_to_model.pkl 
#  --forget_days 62  # only for our dataset
#  --hyp_top_issues 100  # only for our dataset
#  --hyp_top_reports 100  # only for our dataset
```

The script will produce some state files in `event_state` directory in one level with `src`.
These precomputed states will speed up data reading in the next runs.

### Aggregation data collection

To collect data for aggregation training and testing, we need to run this script

```bash
python src/similarity/scripts/similarity.py 
  --data_name netbeans 
  --actions_file path_to_events.csv 
  --reports_path reports_dir 
  --data_start 4340 
  --data_longitude 700 
  --model_path path_to_model.pkl
  --dump_save_path test_dump_path.json
#  --forget_days 62  # only for our dataset
```

The example above show how to collect test data for aggregation model evaluation.

### Aggregation model training

When train and test data for aggregation model were collected, we can train it by running the following script:
```bash
python src/aggregation/scripts/aggregation.py 
  --train_path train_dump.json 
  --test_path test_dump.json
  --model_save_path path_to_aggregation_model.pkl
```
