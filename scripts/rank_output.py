"""This script loads all the model responses with the reference questions and answers,
The user is asked to rank the responses from best to worst, without being shown the model name.
The results are saved to a csv file.
"""
import argparse
import datetime
import logging
import sys
from pathlib import Path
from random import shuffle

import pandas as pd
from columnar import columnar  # type: ignore


def get_user_rankings(prompt_text: str, n_responses: int) -> tuple[list[int], list[int]]:
    print(prompt_text)  # noqa: T201
    rankings: list[int] = []
    scores: list[int] = []
    while True:
        try:
            rankings_in = input("Rankings: ").split()
            rankings = [int(r) for r in rankings_in]
            if len(rankings) != n_responses:
                raise ValueError
            # ensure the numbers are unique and in the range 1 to n_responses
            if len(set(rankings)) != n_responses or not all(1 <= r <= n_responses for r in rankings):
                raise ValueError
            break
        except ValueError:
            print(f"Please enter {n_responses} unique rankings between 1 and {n_responses}")  # noqa: T201
    while True:
        try:
            scores_in = input("Scores: ").split()
            scores = [int(s) for s in scores_in]
            if len(scores) != n_responses:
                raise ValueError
            # ensure the numbers are between 0 and 10
            if not all(0 <= s <= 10 for s in scores):  # noqa: PLR2004
                raise ValueError
            break
        except ValueError:
            print(f"Please enter {n_responses} scores between 0 and 10")  # noqa: T201
    return rankings, scores

if __name__ == "__main__":
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default="reports/report_2024-01-24_094330_cleaned.csv")
    parser.add_argument("--answered", type=Path)
    args = parser.parse_args()
    report_file: Path = args.report
    # open the report
    if not report_file.exists():
        msg = f"Report file {report_file} does not exist"
        raise FileNotFoundError(msg)
    elif report_file.suffix != ".csv":
        msg = f"Report file {report_file} is not a csv file"
        raise ValueError(msg)
    elif report_file.is_dir():
        msg = f"Report file {report_file} is a directory"
        raise ValueError(msg)
    logging.info(f"Loading report from {report_file}")
    report = pd.read_csv(report_file)

    # cols:model,dataset,model_config,command_kw,documents,full_context,question,reference_answer,response,stdout,stderr
    # we only need model, dataset, question, reference_answer, response
    # as each model gave 3 responses to the same prompt, we need to add a response_index column

    # add a response_index column
    report["response_index"] = report.groupby(["model", "dataset", "question"]).cumcount()  # type: ignore

    # get the model, dataset, question, reference_answer, response, response_index columns
    data = report[["model", "dataset", "question", "reference_answer", "response", "response_index"]]

    model_names = list(data["model"].unique())

    # make the answers into columns, e.g.:
    # dataset,question,reference_answer,model_1,model_2,model_3 and a row for each response_index
    data = data.pivot_table(  # type: ignore
        index=["dataset", "question", "reference_answer", "response_index"],
        columns="model",
        values="response",
        aggfunc=lambda x: x,
    ).reset_index()
    total_responses = len(data)
    logging.info(f"Loaded {total_responses} responses from {len(model_names)} models")

    responses: list[dict] = []

    if args.answered:
        # filter out previous responses
        answer_file: Path = args.answered
        if not answer_file.exists():
            msg = f"Continue file {answer_file} does not exist"
            raise FileNotFoundError(msg)
        elif answer_file.suffix != ".csv":
            msg = f"Continue file {answer_file} is not a csv file"
            raise ValueError(msg)
        elif answer_file.is_dir():
            msg = f"Continue file {answer_file} is a directory"
            raise ValueError(msg)
        logging.info(f"Loading continue file from {answer_file}")
        continue_df = pd.read_csv(answer_file)
        total_ranked = len(continue_df)
        logging.info(f"Loaded {total_ranked} responses from {len(model_names)} models")
        # we can exclude responses that match the dataset, question, response_index (reference answer is redundant)
        k = ["dataset", "question", "response_index"]
        i1 = data.set_index(k).index
        i2 = continue_df.set_index(k).index
        data = data[~i1.isin(i2)]  # type: ignore

        remaining_responses = len(data)
        logging.info(f"Excluded {total_responses - remaining_responses} responses ({remaining_responses} remaining)")
        if not remaining_responses == total_responses - total_ranked:
            msg = "Something went wrong with the filtering"
            raise ValueError(msg)
        if remaining_responses == 0:
            logging.warning("You have already ranked all the responses!")
            sys.exit(0)
        # add the responses from the continue file
        responses.extend(continue_df.to_dict(orient="records"))

    question_headers = ["Question", "Reference Answer"]
    response_headers = ["Response " + str(n) for n in range(1, len(model_names) + 1)]

    try:
        # randomize data
        data = data.sample(frac=1).reset_index(drop=True)
        # loop through the responses and ask the user to rank them and give a score from 0 to 10
        for row in data.to_dict(orient="records"):
            response = {
                "dataset": row["dataset"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "response_index": row["response_index"]
            }
            # get random model sequence
            shuffle(model_names)
            # display two columns, one with the question and one with the reference answer
            table = columnar(
                [[row["question"], row["reference_answer"]]],
                question_headers,
                no_borders=False,
                wrap_max=100,
            )
            print(table)  # noqa: T201
            # then display a column for each model response
            table = columnar(
                [[row[model_name] for model_name in model_names]],
                response_headers,
                no_borders=False,
                wrap_max=100,
            )
            print(table)  # noqa: T201
            # ask the user to rank the responses
            prompt_text = "Rank the responses from best to worst response, using the response number, e.g. 4 2 3 1\n"
            prompt_text += "Give a score from 0 to 10 for each response, 0 being the worst and 10 being the best for\n"
            prompt_text += "each response (left to right e.g. 10 5 7 2)\n"
            rankings, scores = get_user_rankings(prompt_text=prompt_text, n_responses=len(model_names))
            # add a ranking and score for each model
            for model_name, score in zip(model_names, scores, strict=True):
                response[model_name + "_rank"] = rankings.index(model_names.index(model_name) + 1) + 1
                response[model_name + "_score"] = score
            responses.append(response)
            print()  # noqa: T201
    except Exception as e:
        logging.error("Something went wrong!")
        logging.error(e)
    finally:
        # save the results to a csv file
        # add a timestamp to the filename
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H%M%S")
        logging.info(f"Saving results to reports/rankings_{timestamp}.csv")
        df = pd.DataFrame(responses)
        df.to_csv(f"reports/rankings_{timestamp}.csv", index=False)

    logging.info("Thank you!")
