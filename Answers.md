1> Are there unit tests for the API?
Yes, in folder unittests -> ApiTests.py
2> Are there unit tests for the model?
Yes, in folder unittests -> ModelTests.py
3> Are there unit tests for the logging?
Yes, in folder unittests -> LoggerTests.py
4> Can all of the unit tests be run with a single script and do all of the unit tests pass?
Yes -> run-tests.py
5> Is there a mechanism to monitor performance?
Yes -> monitoring.py
6> Was there an attempt to isolate the read/write unit tests from production models and logs?
Yes -> model.py & logger.py
7> Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?
Sometimes
8> Does the data ingestion exists as a function or script to facilitate automation?
script
9> Were multiple models compared?
Yes -> run-model-train.py
10> Did the EDA investigation use visualizations?
Yes
11> Is everything containerized within a working Docker image?
Yes -> dockerfile
12> Did they use a visualization to compare their model to the baseline model?
No