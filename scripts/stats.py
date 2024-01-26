"""Get statistics for generation times for different models."""

import matplotlib.pyplot as plt
import pandas as pd

# open the report
report = pd.read_csv("reports/report_2024-01-24_094330_cleaned.csv")

# cols:
# model,dataset,model_config,command_kw,documents,full_context,question,reference_answer,response,stdout,stderr
# model is the model used, dataset is the collection of markdown files, Stderr contains the llama_print_timings outputs
# example: (but stderr contains other outputs as well)
# llama_print_timings:        load time =    2651.32 ms
# llama_print_timings:      sample time =      28.45 ms /   174 runs   (    0.16 ms per token,  6117.07 tokens per second)
# llama_print_timings: prompt eval time =   36205.82 ms /  1181 tokens (   30.66 ms per token,    32.62 tokens per second)
# llama_print_timings:        eval time =   10031.68 ms /   173 runs   (   57.99 ms per token,    17.25 tokens per second)
# llama_print_timings:       total time =   46640.01 ms

# get the timings, we want load, sample, prompt eval, eval, total
# for sample, prompt and eval we want the tokens per second
timings = report["stderr"].str.extract(
    r"[.\n]*load[^=]+=[^\d]+([\d\.]+).*"
    r"[.\n]*.*sample[^=]+=[^\d]+([\d\.]+)[^\d]+(\d+)[^\d]+([\d\.]+)[^\d]+([\d\.]+).*"
    r"[.\n]*.*prompt[^=]+=[^\d]+([\d\.]+)[^\d]+(\d+)[^\d]+([\d\.]+)[^\d]+([\d\.]+).*"
    r"[.\n]*.*eval[^=]+=[^\d]+([\d\.]+)[^\d]+(\d+)[^\d]+([\d\.]+)[^\d]+([\d\.]+).*"
    r"[.\n]*.*total[^=]+=[^\d]+([\d\.]+).*"
)
print(timings)

# rename the columns
timings.columns = [
    "load_time",
    "sample_time",
    "sample_runs",
    "sample_ms_per_token",
    "sample_tokens_per_second",
    "prompt_eval_time",
    "prompt_eval_tokens",
    "prompt_eval_ms_per_token",
    "prompt_eval_tokens_per_second",
    "eval_time",
    "eval_runs",
    "eval_ms_per_token",
    "eval_tokens_per_second",
    "total_time",
]


# convert to numeric
timings = timings.apply(pd.to_numeric)

# add the model and dataset columns
timings["model"] = report["model"]
timings["dataset"] = report["dataset"]

# save the timings
timings.to_csv("reports/timings.csv", index=False)

# prompt_eval_tokens_per_second, we need to exclude when reusing a prompt
# so prompt_eval_time != 0

# get the mean and std of the prompt_eval_tokens_per_second for each model
mean_prompt_eval_tokens_per_second = timings[
    timings["prompt_eval_time"] != 0
].groupby("model")["prompt_eval_tokens_per_second"].mean()
std_prompt_eval_tokens_per_second = timings[
    timings["prompt_eval_time"] != 0
].groupby("model")["prompt_eval_tokens_per_second"].std()


# get the mean and std of the eval_tokens_per_second for each model
mean_eval_tokens_per_second = timings.groupby("model")["eval_tokens_per_second"].mean()
std_eval_tokens_per_second = timings.groupby("model")["eval_tokens_per_second"].std()

# get the mean and std load time for each model
mean_load_time = timings.groupby("model")["load_time"].mean()
std_load_time = timings.groupby("model")["load_time"].std()

# make a plot with error bars of the means and std per model
# subplot per value
# highlight the best model per value
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].errorbar(
    mean_prompt_eval_tokens_per_second.index,
    mean_prompt_eval_tokens_per_second,
    yerr=std_prompt_eval_tokens_per_second,
    fmt="o",
)
axs[0].set_title("prompt_eval_tokens_per_second")
axs[0].set_ylabel("tokens per second")
axs[0].set_xlabel("model")
axs[0].tick_params(axis="x", rotation=90)
axs[1].errorbar(
    mean_eval_tokens_per_second.index,
    mean_eval_tokens_per_second,
    yerr=std_eval_tokens_per_second,
    fmt="o",
)
axs[1].set_title("eval_tokens_per_second")
axs[1].set_ylabel("tokens per second")
axs[1].set_xlabel("model")
axs[1].tick_params(axis="x", rotation=90)
axs[2].errorbar(
    mean_load_time.index, mean_load_time, yerr=std_load_time, fmt="o",
)
axs[2].set_title("load_time")
axs[2].set_ylabel("time (ms)")
axs[2].set_xlabel("model")
axs[2].tick_params(axis="x", rotation=90)

# highlight the best model per value
best_prompt_eval_tokens_per_second = mean_prompt_eval_tokens_per_second.idxmax()
best_eval_tokens_per_second = mean_eval_tokens_per_second.idxmax()
best_load_time = mean_load_time.idxmin()
axs[0].scatter(best_prompt_eval_tokens_per_second, mean_prompt_eval_tokens_per_second[best_prompt_eval_tokens_per_second], marker="x", s=100, c="r")
axs[1].scatter(best_eval_tokens_per_second, mean_eval_tokens_per_second[best_eval_tokens_per_second], marker="x", s=100, c="r")
axs[2].scatter(best_load_time, mean_load_time[best_load_time], marker="x", s=100, c="r")
fig.tight_layout()
fig.savefig("reports/timings.png")
plt.close(fig)








