# %%
import logging
import pandas as pd
import numpy as np
import os
import json
import re

logger = logging.getLogger(__name__)


# %%
def extract_valid_json(string: str) -> dict:
    """Extract and validate JSON answer from LLM response string."""
    # print(f'> > EXTRACT VALID JSON: input: "{string}"')
    string_clean = string.replace("\n", "")
    # print(f"\t{string_clean=}\n")
    # Regex pattern for basic JSON structure: objects {} and arrays []
    json_pattern = re.compile(r"\{.*?\}|\[.*?\]", re.DOTALL)
    # Finding all matches that look like JSON
    potential_jsons = json_pattern.findall(string_clean)
    # print(f"\t{potential_jsons=}\n")
    if not potential_jsons:
        return None
    potential_jsons.reverse()
    for pj in potential_jsons:
        try:
            # Attempt to parse the JSON
            valid_json = json.loads(pj)
            # Returning the first valid JSON found
            if isinstance(valid_json, dict):
                valid_json = {k.upper(): v for k, v in valid_json.items()}
                if "ANSWER" in valid_json.keys():
                    answer_val = valid_json["ANSWER"]
                    if isinstance(answer_val, str) and answer_val in [
                        "A",
                        "B",
                        "C",
                        "D",
                    ]:
                        return valid_json
        except json.JSONDecodeError:
            continue
    return None


def model_clean(model_str: str) -> str:
    """Clean model name for filesystem usage."""
    model_str = model_str.split("/")[-1]
    model_str = model_str.replace(".", "_").replace(":", "_")
    return model_str


def score_and_save(
    cfg,
    questions: list[dict],
    round_num: int,
    omit_models: list[str] = None,
    models_override: dict = None,
) -> dict[str, pd.DataFrame]:
    """Score model answers and save results to JSON files."""
    omit_models = omit_models or []
    answers_save_path = f"{cfg.SaveFolders.answers_save_path}/round_{round_num}"
    os.makedirs(answers_save_path, exist_ok=True)
    all_answers_df = {}
    models = models_override if models_override is not None else cfg.models
    for model, _model_params in models.items():
        if model in omit_models:
            continue
        final_answers = []
        for question in questions:
            answer_block = {k: v for k, v in question.items() if k != "responses"}
            janswer = question["responses"][model + "_choice"]
            janswer = janswer if janswer else {"ANSWER": ""}
            answer_block["model_answer"] = question["responses"][model]
            answer_block["json_answer"] = janswer
            answer_block["json_answer_letter"] = janswer["ANSWER"]
            answer_block["invalid_answer_letter"] = int(
                janswer["ANSWER"] not in ["A", "B", "C", "D"]
            )
            answer_block["score"] = (
                100 if janswer["ANSWER"] == answer_block["correct_letter"] else 0
            )
            final_answers.append(answer_block)
        answers_df = pd.DataFrame(final_answers)
        if "index" not in answers_df.columns:
            answers_df.reset_index(inplace=True)
        answers_df.set_index("index").to_json(
            f"{answers_save_path}/final_answers-{model}.json",
            orient="records",
            lines=True,
        )
        all_answers_df[model] = answers_df
    return all_answers_df


def load_all_llm_answers_from_json(
    answers_save_path: str,
    prefix_replace="final_answers-",
    sub_folders=None,
) -> dict[pd.DataFrame]:
    """Reload all scored answers from JSON files."""
    if sub_folders is None:
        sub_folders = [""]
    all_llm_answers = {}
    for sub_folder in sub_folders:
        answers_save_path_sub = f"{answers_save_path}{sub_folder}"
        if not os.path.exists(answers_save_path_sub):
            continue
        for output_file in os.listdir(f"{answers_save_path_sub}/"):
            if output_file.endswith(".json"):
                outputs_df = pd.read_json(
                    f"{answers_save_path_sub}/{output_file}",
                    orient="records",
                    lines=True,
                )
                model = output_file.replace(prefix_replace, "").replace(".json", "")
                all_llm_answers.setdefault(model, pd.DataFrame())
                all_llm_answers[model] = pd.concat(
                    [all_llm_answers[model], outputs_df]
                ).reset_index()
    return all_llm_answers


def load_all_answers_by_round(
    save_parent: str, agent_name: str = None, num_rounds: int = 10
) -> dict[str, list[pd.DataFrame]]:
    """Load all answers organized by round."""
    all_results: dict[str, list[pd.DataFrame]] = {}
    for n in range(num_rounds):
        # print(f"\n----- Round: {n + 1} of {num_rounds} -----")
        answer_save_path_round = f"{save_parent}/round_{n + 1}"
        round_results = load_all_llm_answers_from_json(
            answer_save_path_round, prefix_replace="final_answers-"
        )
        if agent_name is not None:
            round_results = {agent_name: round_results.get(agent_name)}
        for agent, results in round_results.items():
            if agent not in all_results:
                all_results[agent] = []
            all_results[agent].append(results)

    return all_results


def calculate_llm_stats(
    cfg, all_llm_answers: dict[str, pd.DataFrame], bootstrap_n=10000
) -> dict:
    """Calculate statistics for LLM performance with bootstrap CI."""
    all_llm_stats = {}
    for model, outputs in all_llm_answers.items():
        logger.info("Calculating stats for %s: #%d outputs", model, len(outputs))
        mean_score = outputs["score"].mean()
        std_dev_score = outputs["score"].std()
        # Vectorized bootstrap for 95% CI
        scores = outputs["score"].values
        samples = np.random.choice(
            scores, size=(bootstrap_n, len(scores)), replace=True
        )
        bootstrap_means = samples.mean(axis=1)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        # caculate z-interval 95%
        z = 1.96
        z_interval_error = z * (std_dev_score / np.sqrt(len(outputs)))
        all_llm_stats[model] = {
            "mean_score": mean_score,
            "std_dev_score": std_dev_score,
            "z_interval_error": z_interval_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "output_count": len(outputs),
        }
    return all_llm_stats


def get_llm_stats(
    cfg, all_llm_answers: list[list[pd.DataFrame]], file_suffix="", bootstrap_n=10000
) -> pd.DataFrame:
    all_llm_stats = calculate_llm_stats(cfg, all_llm_answers, bootstrap_n)
    try:
        stats_df = (
            pd.DataFrame(all_llm_stats)
            .transpose()
            .sort_values("mean_score", ascending=False)
        )
    except KeyError as e:
        logger.error("No stats to calculate: %s", e)
        exit()
    stats_df.index.name = "model"
    os.makedirs(cfg.SaveFolders.stats_save_path, exist_ok=True)
    stats_df.to_csv(f"{cfg.SaveFolders.stats_save_path}/final_stats{file_suffix}.csv")
    return stats_df


def report_failures(answers_save_path: str, agent_name: str = None):
    max_idx = 30  # number of total questions
    results = load_all_answers_by_round(answers_save_path, agent_name)
    failures = {k: [] for k in results}
    fail_index_counts = {k: {j: 0 for j in range(max_idx)} for k in results}
    for agent, results_list in results.items():
        for round_results in results_list:
            for idx, res in round_results.iterrows():
                if res["score"] == 0:
                    failures[agent].append(res)
                    fail_index_counts[agent][idx] += 1

    for agent, fail_list in fail_index_counts.items():
        filter_fail_list = {k: n for k, n in fail_list.items() if n >= 7}
        if filter_fail_list:
            logger.info("***** %s *****", agent)
            for k, n in filter_fail_list.items():
                logger.info("Question %d failed %d times", k, n)
            logger.info("***** ***** *****")

    os.makedirs("./reports", exist_ok=True)
    for agent, fail_list in failures.items():
        df = pd.DataFrame(fail_list)
        df.to_html(f"./reports/{agent}.html")
    return
