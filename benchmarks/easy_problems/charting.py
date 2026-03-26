import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def define_data(final_stats: pd.DataFrame):
    """Prepare data for charting by adding human baseline."""
    # # Define the data
    # models = ["Human level*", "GPT-4 Turbo", "Claude 3 Opus", "Mistral Large", "Gemini Pro 1.5",
    #           "Gemini Pro 1.0", "Llama 3 70B", "Mistral 8x22B"]
    # mean_scores = [80, 38, 33, 30, 29, 27, 21, 16]
    # lower_bounds = [10, 16, 15, 15, 15, 15, 13, 11]
    # upper_bounds = [10, 16, 15, 15, 15, 15, 13, 11]
    final_stats.loc[-1] = {
        "model": "Human level*",
        "mean_score": 86,
        "std_dev_score": 0,
        "z_interval_error": 0,
        "ci_lower": 78,
        "ci_upper": 93,
    }
    final_stats = final_stats.sort_values(by="mean_score", ascending=False)

    models = final_stats["model"].to_list()
    mean_scores = final_stats["mean_score"].to_list()
    lower_bounds = final_stats["ci_lower"].to_list()
    upper_bounds = final_stats["ci_upper"].to_list()

    data = {
        "Model": models,
        "Average": mean_scores,
        "Confidence Interval Low": lower_bounds,
        "Confidence Interval High": upper_bounds,
    }
    return pd.DataFrame(data)


def create_performance_chart(
    final_stats: pd.DataFrame,
    title="LLM Linguistic Benchmark Performance",
):
    """Create a performance chart with confidence intervals."""

    df = define_data(final_stats)
    # Create a basic barplot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    barplot = sns.barplot(data=df, x="Model", y="Average", errorbar=None)

    # Shade the first bar with black cross lines
    for i, bar in enumerate(barplot.patches):  # Loop through the bars
        if df["Model"][i] == "Human level*":  # Check if it is the Human level bar
            bar.set_hatch("///")  # Apply hatching

    # Add confidence intervals as vertical lines with caps
    capwidth = 0.1  # Width of the cap lines
    for i, model in enumerate(df["Model"]):
        plt.plot(
            [i, i],
            [df["Confidence Interval Low"][i], df["Confidence Interval High"][i]],
            color="grey",
            lw=1,
        )
        # Add horizontal caps
        plt.plot(
            [i - capwidth / 2, i + capwidth / 2],
            [df["Confidence Interval Low"][i], df["Confidence Interval Low"][i]],
            color="grey",
            lw=1,
        )
        plt.plot(
            [i - capwidth / 2, i + capwidth / 2],
            [df["Confidence Interval High"][i], df["Confidence Interval High"][i]],
            color="grey",
            lw=1,
        )

    plt.title(title, fontsize=18)
    plt.xlabel("", fontsize=14)
    plt.ylabel("Average Score (%)", fontsize=14)
    plt.xticks(rotation=60, fontsize=14)
    plt.tight_layout()

    return barplot, plt
