"""This script loads all the model responses with the reference questions and answers,
The user is asked to rank the responses from best to worst, without being shown the model name.
The results are saved to a csv file.
"""
from random import shuffle

import pandas as pd
from columnar import columnar

# open the report
report = pd.read_csv("reports/report_2024-01-24_094330_cleaned.csv")

# cols: model,dataset,model_config,command_kw,documents,full_context,question,reference_answer,response,stdout,stderr
# we only need model, dataset, question, reference_answer, response
# as each model gave 3 responses to the same prompt, we need to add a response_index column

# add a response_index column
report["response_index"] = report.groupby(["model", "dataset", "question"]).cumcount()

# get the model, dataset, question, reference_answer, response, response_index columns
data = report[["model", "dataset", "question", "reference_answer", "response", "response_index"]]

model_names = list(data["model"].unique())

# make the answers into columns, e.g.:
# dataset,question,reference_answer,model_1,model_2,model_3 and a row for each response_index
data = data.pivot_table(
    index=["dataset", "question", "reference_answer", "response_index"],
    columns="model",
    values="response",
    aggfunc=lambda x: x,
).reset_index()

# randomise the order of the responses
data = data.sample(frac=1).reset_index(drop=True)

question_headers = ["Question", "Reference Answer"]
response_headers = ["Response " + str(n) for n in range(1, len(model_names) + 1)]
# loop through the responses and ask the user to rank them and give a score from 0 to 10
for i, row in data.iterrows():
    # get random model sequence
    shuffle(model_names)
    # display two columns, one with the question and one with the reference answer
    table = columnar(
        [[row["question"], row["reference_answer"]]],
        question_headers,
        no_borders=False
    )
    print(table)
    # then display a column for each model response
    table = columnar(
        [[row[model_name]] for model_name in model_names],
        response_headers,
        no_borders=False
    )
    print(table)
    # ask the user to rank the responses
    print("Rank the responses from best to worst")
    break