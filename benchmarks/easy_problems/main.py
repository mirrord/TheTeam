from .charting import create_performance_chart
from .eval_utils import (
    get_llm_stats,
    load_all_llm_answers_from_json,
    model_clean,
    score_and_save,
    extract_valid_json,
    report_failures,
)

from copy import deepcopy
from .llm_service import (
    ollama_llm_service,
    ollama_reflector,
    ollama_reflector_adversarial,
    ollama_reflector_simple,
    ollama_reflector_adversarial_pointed,
    ollama_stepper,
    ollama_named,
    ollama_flow,
    dummy_llm_service,
)
from .multiple_choice import construct_multiple_choice_question
from .benchmark_config import BenchmarkConfig
from .dataset import load_dataset
from .cli import (
    parse_args,
    apply_cli_overrides,
    list_available_configs,
    list_available_datasets,
)
import logging
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib
import os
from pithos import ConfigManager

logger = logging.getLogger(__name__)


def myprogress(curr, N, width=10, bars="▉▊▋▌▍▎▏ "[::-1], full="█", empty=" "):
    """Generate a progress bar string."""
    p = curr / N
    nfull = int(p * width)
    bar_idx = int(len(bars) * ((p * width) % 1))
    return "{:>3.0%} |{}{}{}| {:>2}/{}".format(
        p, full * nfull, bars[bar_idx], empty * (width - nfull - 1), curr, N
    )


def load_benchmark_questions(dataset_path: str = None):
    """Load benchmark questions from dataset file.

    Args:
        dataset_path: Optional path to dataset. If None, uses default.

    Returns:
        List of question dictionaries
    """
    if dataset_path is None:
        # Default to the linguistic benchmark in the same directory
        cwd = pathlib.Path(__file__).parent.resolve()
        dataset_path = f"{cwd}/linguistic_benchmark_multi_choice.json"

    # Use dataset abstraction layer
    dataset = load_dataset("multiple_choice", dataset_path)
    questions = dataset.get_questions()

    # Convert to dictionary format for backward compatibility
    return [q.to_dict() for q in questions]


def copy_question(q: dict, prompt: str, correct_letter: str) -> dict:
    """Create a copy of a question with updated fields."""
    newq = deepcopy(q)
    newq.update(
        {
            "multi_choice_question": prompt,
            "correct_letter": correct_letter,
            "responses": {},
        }
    )
    return newq


# returns all_llm_answers: dict[model : list[question dataframe]]
def get_all_llm_answers(cfg, models: dict) -> dict[str, pd.DataFrame]:
    logger.info("GETTING LLM ANSWERS")
    num_rounds = cfg.answer_rounds
    answers_save_path = cfg.SaveFolders.answers_save_path

    # Load benchmark questions using dataset path from config
    dataset_path = (
        getattr(cfg.dataset, "path", None) if hasattr(cfg, "dataset") else None
    )
    benchmark_questions = load_benchmark_questions(dataset_path)

    all_scores_df = {k: [] for k in models.keys()}
    for n in range(num_rounds):
        logger.info("Round: %d of %d", n + 1, num_rounds)
        answer_save_path_round = f"{answers_save_path}/round_{n + 1}"
        round_results = load_all_llm_answers_from_json(
            answer_save_path_round, prefix_replace="final_answers-"
        )

        already_run_models = list(round_results.keys())
        models_to_run = [model for model in models if model not in round_results.keys()]
        round_results.update({model: pd.DataFrame() for model in models_to_run})

        if models_to_run:
            mc_questions = {
                qidx: construct_multiple_choice_question(question)
                for qidx, question in enumerate(benchmark_questions)
            }
            questions = [
                copy_question(benchmark_questions[qidx], prompt, answer)
                for qidx, (prompt, answer) in mc_questions.items()
            ]

            def _run_model(model_name):
                """Run all questions for a single model, returning per-question results."""
                model_questions = deepcopy(questions)
                num_retries = cfg.answer_hyperparams["num_retries"]
                for qidx, question in enumerate(model_questions):
                    answer = None
                    rsp = ""
                    for i in range(num_retries + 1):
                        service_func = models[model_name]["service_function"]
                        rsp = service_func.completion(question["multi_choice_question"])
                        answer = extract_valid_json(rsp)
                        if answer is not None:
                            break
                        logger.warning(
                            "Invalid response for model %s question %d (retry %d): %s",
                            model_name,
                            qidx,
                            i + 1,
                            rsp,
                        )
                        if i < num_retries:
                            time.sleep(2**i)
                    if answer is None:
                        answer = {"ANSWER": ""}
                    model_questions[qidx]["responses"][model_name] = rsp
                    model_questions[qidx]["responses"][f"{model_name}_choice"] = answer
                return model_name, model_questions

            with ThreadPoolExecutor(max_workers=len(models_to_run)) as executor:
                futures = {
                    executor.submit(_run_model, model_name): model_name
                    for model_name in models_to_run
                }
                for future in as_completed(futures):
                    model_name, model_questions = future.result()
                    # Per-model checkpoint: score and save immediately
                    model_round = score_and_save(
                        cfg,
                        model_questions,
                        round_num=n + 1,
                        omit_models=[m for m in models if m != model_name]
                        + already_run_models,
                        models_override=models,
                    )
                    round_results.update(model_round)
                    logger.info("Checkpoint saved: %s round %d", model_name, n + 1)

        for model, answers_df in round_results.items():
            if model not in all_scores_df:
                all_scores_df[model] = [answers_df]
            else:
                all_scores_df[model].append(answers_df)

    for model, evals_df in all_scores_df.items():
        logger.info("Model: %s, %d Rounds collected", model, len(evals_df))
        for i, eval_df in enumerate(evals_df):
            logger.debug("Round %d: %d questions answered", i + 1, len(eval_df))

    logger.info("-- ANSWERS COMPLETE --")
    return {m: pd.concat(v).reset_index() for m, v in all_scores_df.items()}


def generate_statistics(cfg, all_llm_evals: dict[str, list[pd.DataFrame]]):
    logger.info("GENERATING STATISTICS")

    all_stats_dfs = {}

    chart_title = "LLM Linguistic Benchmark Performance"
    stats_df = get_llm_stats(
        cfg,
        all_llm_evals,
    )

    for model, evals_df in all_llm_evals.items():
        logger.info("Model: %s", model)
        # evals_df['invalid_answer_letter'] = evals_df.apply(lambda x: x['json_answer_letter'] not in ['A', 'B', 'C', 'D'], axis=1)
        incorrect_letter_count = evals_df["invalid_answer_letter"].sum()
        # print(model, incorrect_letter_count)
        stats_df.loc[model, "invalid_outputs"] = incorrect_letter_count

        auto_eval_agg = (
            all_llm_evals[model].groupby("index").agg({"score": ["mean", "min", "max"]})
        )
        auto_eval_agg.index.name = "Question #"
        logger.debug("\n%s", auto_eval_agg)

    # Only create charts if configured to do so
    create_charts = (
        getattr(cfg.output, "create_charts", True) if hasattr(cfg, "output") else True
    )
    if create_charts:
        barplot, plt = create_performance_chart(
            stats_df.reset_index(),
            chart_title,
        )
        chart_location = f"{cfg.SaveFolders.stats_save_path}/performance_chart.png"
        barplot.figure.savefig(chart_location)
        logger.info("Performance chart saved to %s", chart_location)
        # plt.show() #UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown

    # print("final stats table:")
    # print(stats_df)
    all_stats_dfs[chart_title] = stats_df
    logger.info("-- STATS COMPLETE --")
    return all_stats_dfs


# -------


def run_all_steps(cfg, models: dict):
    os.makedirs("./results", exist_ok=True)
    answers = get_all_llm_answers(cfg, models)
    return generate_statistics(cfg, answers)


def target_save_path(cfg, params: list) -> str:
    base_path = cfg.SaveFolders.answers_save_path
    base_split = base_path.split("-")
    base_date = base_split[:3]
    base_label = "-".join(base_split[3:])
    for idx, p in enumerate(params):
        base_date[2 - idx] = p
    return "-".join(base_date) + "-" + base_label


def main():
    """Main entry point for benchmark CLI."""
    # Parse CLI arguments
    args = parse_args()

    # Handle list commands
    if args.command == "list-configs":
        list_available_configs()
        return

    if args.command == "list-datasets":
        list_available_datasets()
        return

    # Handle report command
    if args.command == "report":
        if args.path:
            save_path = args.path
        elif hasattr(args, "date") and args.date:
            # Construct path from date
            save_path = f"./results/{args.date}-Multi-Benchmark/llm_outputs"
        else:
            # Try to load config to get default path
            cfg = BenchmarkConfig.from_yaml_or_default(getattr(args, "config", None))
            save_path = cfg.SaveFolders.answers_save_path

        logger.info("Creating report from test: %s", save_path)
        report_failures(save_path)
        return

    # Handle run command (default)
    cfg_manager = ConfigManager()

    # Load benchmark configuration
    cfg = BenchmarkConfig.from_yaml_or_default(getattr(args, "config", None))

    # Apply CLI overrides if running with new config
    if hasattr(args, "models"):
        cfg = apply_cli_overrides(cfg, args)

    logger.info("Running benchmark: %s", cfg.name)
    logger.info("Models: %s", list(cfg.models.keys()))
    logger.info("Rounds: %d", cfg.execution.rounds)
    logger.info("Dataset: %s", cfg.dataset.path)
    logger.info("Output: %s", cfg.folder_name)
    print()

    # Set up service functions (same as before)
    service_funcs = {
        "dummy": dummy_llm_service,
        "ollama": ollama_llm_service,
        "ollama_reflector": ollama_reflector,
        "ollama_adversarial": ollama_reflector_adversarial,
        "ollama_reflector_simple": ollama_reflector_simple,
        "ollama_adversarial_pointed": ollama_reflector_adversarial_pointed,
        "ollama_stepper": ollama_stepper,
        "ollama_named": lambda c: ollama_named(c, cfg_manager),
    }

    # Build working models dict with cleaned names — leave cfg.models untouched
    working_models = {
        model_clean(model_name): deepcopy(model_dict)
        for model_name, model_dict in cfg.models.items()
    }

    # Validate and establish service functions for each model
    for model_name, model_dict in working_models.items():
        if model_dict["service"] == "ollama_flow":
            flowchart_name = model_dict.get("flowchart")
            if not flowchart_name:
                raise ValueError(
                    f"Model '{model_name}' uses ollama_flow but has no 'flowchart' key"
                )
            llm_service_func = ollama_flow(
                model_dict["model"], flowchart_name, cfg_manager
            )
        else:
            service_cls = service_funcs.get(model_dict["service"])
            if service_cls is None:
                raise ValueError(
                    f"Unknown service type '{model_dict['service']}' for model '{model_name}'. "
                    f"Valid types: {list(service_funcs.keys()) + ['ollama_flow']}"
                )
            llm_service_func = service_cls(model_dict["model"])
        working_models[model_name]["service_function"] = llm_service_func

    # Run the benchmark
    run_all_steps(cfg, working_models)


if __name__ == "__main__":
    main()
