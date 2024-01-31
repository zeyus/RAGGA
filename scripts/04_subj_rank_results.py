"""Plot and save the results of the subjective response ranking."""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker


def __chain_flatten_cols(self, how="_".join, reset_index=True):  # noqa: FBT002
    how = (lambda itr: list(itr)[-1]) if how == "last" else how
    self.columns = [
            how(
                filter(
                    None,
                    map(
                        str,
                        levels
                    )
                )
            ) for levels in self.columns.values  # type: ignore
        ] if isinstance(self.columns, pd.MultiIndex) else self.columns
    return self.reset_index() if reset_index else self
pd.DataFrame.chain_flatten_cols = __chain_flatten_cols  # type: ignore

if __name__ == "__main__":
    # set figure size
    fig_height = 8
    fig_width = 12
    fig_aspect = fig_width / fig_height
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)  # type: ignore
    # set dpi
    plt.rcParams["figure.dpi"] = 300  # type: ignore
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default="reports/rankings_2024-01-30_195244.csv")
    args = parser.parse_args()
    rankings_file: Path = args.rankings
    logging.info(f"Using rankings file: {rankings_file}")
    # open the rankings
    rankings = pd.read_csv(rankings_file)
    # get the model names
    # each model has a _score and a _rank column
    # let's make it long format
    rankings_long = rankings.melt(id_vars=["dataset", "question", "reference_answer", "response_index"])  # type: ignore
    # get the model names

    # split _rank and _score into separate columns
    rankings_long["type"] = rankings_long["variable"].str.split("_").str[-1]
    rankings_long["model"] = rankings_long["variable"].str.split("_").str[:-1].str.join("_")
    model_names = rankings_long["model"].unique()
    # drop the variable column
    rankings_long = rankings_long.drop(columns=["variable"])
    # plot the rankings
    fig, ax = plt.subplots()
    # plot the rankings
    sns.boxplot(
        data=rankings_long.query("type == 'rank'"),
        x="model",
        y="value",
        fill=False,
        ax=ax,
        gap=0.5,
        hue="model",
    )
    # add scatter plot with jitter
    sns.stripplot(
        data=rankings_long.query("type == 'rank'"),
        x="model",
        y="value",
        ax=ax,
        hue="model",
        legend=False,
        alpha=0.3,
    )

    # reverse the y axis
    ax.invert_yaxis()  # type: ignore
    # only show integer ticks on y axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # type: ignore
    # set title
    ax.set_title("Response Rankings by Model (lower values are better)")
    logging.info(f"Model names: {model_names}")
    logging.info("Saving rankings plot to reports/rankings_by_model.png")
    # save the plot
    fig.savefig("reports/rankings_by_model.png")

    fig, ax = plt.subplots()
    # plot the scores by model
    model_names.sort()
    sns.boxplot(
        data=rankings_long.query("type == 'score'").sort_values("model"),
        x="model",
        y="value",
        ax=ax,
        order=model_names,
        fill=False,
        hue="model",
        gap=0.5,
    )
    # add scatter plot with jitter
    sns.stripplot(
        data=rankings_long.query("type == 'score'").sort_values("model"),
        x="model",
        y="value",
        legend=False,
        order=model_names,
        ax=ax,
        hue="model",
        alpha=0.3,
    )
    # set title
    ax.set_title("Response Scores by Model")
    ax.set_ylabel("score")
    # save the plot
    logging.info("Saving scores plot to reports/scores_by_model.png")
    fig.savefig("reports/scores_by_model.png")

    # fig, ax = plt.subplots()
    # add score and rank to columns
    rankings_long["score"] = rankings_long["value"].where(rankings_long["type"] == "score")
    rankings_long["rank"] = rankings_long["value"].where(rankings_long["type"] == "rank")
    # drop the value column
    rankings_long = rankings_long.drop(columns=["value"])
    # merge the rows with the same dataset, question, reference_answer, model and response_index
    rankings_long = rankings_long.groupby(
        [
            "dataset",
            "question",
            "reference_answer",
            "model",
            "response_index"
        ]).sum().reset_index()  # type: ignore

    # # make a plot of total scores
    # rankings_long.groupby("model")[["model","score"]].agg({
    #         "score": ["mean", "std"]
    #     }).chain_flatten_cols().plot(
    #     ax=ax,
    #     yerr="score_std",
    #     kind="scatter",
    #     x="model",
    #     y="score_mean",
    #     rot=0,
    # )
    # # set title
    # ax.set_title("Mean and Standard Deviation of Model Response Scores")
    # # save the plot
    # logging.info("Saving scores plot to reports/mean_scores_by_model.png")
    # fig.savefig("reports/mean_scores.png")

    # # make a plot of total rank
    # fig, ax = plt.subplots()
    # # plot mean and std of total rank
    # rankings_long.groupby("model")[["model","rank"]].agg({
    #         "rank": ["mean", "std"]
    #     }).chain_flatten_cols().plot(
    #     ax=ax,
    #     yerr="rank_std",
    #     kind="scatter",
    #     x="model",
    #     y="rank_mean",
    #     rot=0,
    # )
    # # reverse the y axis
    # ax.invert_yaxis()
    # # set title
    # ax.set_title("Mean and Standard Deviation of Model Response Ranks")
    # # save the plot
    # logging.info("Saving ranks plot to reports/mean_ranks.png")
    # fig.savefig("reports/mean_ranks.png")


    # now we have a score and rank column
    # scatter plots of individal score vs individual ranks colored by model
    fig, ax = plt.subplots()
    # sns.stripplot(
    #     data=rankings_long,
    #     x="rank",
    #     y="score",
    #     hue="model",
    # )
    sns.lmplot(
        data=rankings_long,
        x="score",
        y="rank",
        hue="model",
        x_jitter = 0.2,
        y_jitter=0.2,
        scatter_kws={
            "alpha": 0.3
        },
        height=fig_height,
        aspect=fig_aspect,
    )
    # set limits
    plt.xlim(-0.5, 10.5)
    plt.ylim(0.5, len(model_names) + 0.5)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(1, len(model_names) + 1, 1))

    # set title
    ax.set_title("Response Score vs Response Rank by Model")
    # save the plot
    logging.info("Saving scatter plot to reports/score_vs_rank.png")
    plt.savefig("reports/score_vs_rank.png")


    # make a plot of score by dataset + model
    fig, ax = plt.subplots()
    # plot mean and std of score
    sns.boxplot(
        data=rankings_long,
        x="dataset",
        y="score",
        fill=False,
        hue="model",
        gap=0.5,
    )
    # add scatter plot with jitter
    sns.stripplot(
        data=rankings_long,
        x="dataset",
        y="score",
        hue="model",
        legend=False,
        dodge=True,
        alpha=0.3,
    )
    sns.move_legend(ax, "upper right", ncol=len(model_names), frameon=False, title=None)
    # set title
    ax.set_title("Response Scores by Model and Dataset")
    # save the plot
    logging.info("Saving scores plot to reports/mean_scores_by_dataset.png")
    fig.savefig("reports/mean_scores_by_dataset.png")
