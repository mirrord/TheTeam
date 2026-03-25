# Benchmarking Guide

This guide explains how to use pithos's benchmarking system to evaluate and compare LLM agents and workflows.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Dataset Management](#dataset-management)
- [Creating Custom Benchmarks](#creating-custom-benchmarks)
- [Understanding Results](#understanding-results)

## Installation

The benchmarking system requires additional dependencies beyond the core pithos package:

```bash
# Install pithos with benchmark dependencies
pip install -e ".[benchmark]"
```

This installs:
- `pandas` - Data analysis and manipulation
- `matplotlib` - Chart generation
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars

Alternatively, install them manually:

```bash
pip install pandas matplotlib seaborn tqdm
```

## Quick Start

Run a benchmark with the default configuration:

```bash
pithos-benchmark
```

This will:
1. Load the default benchmark configuration from `configs/benchmarks/default_benchmark.yaml`
2. Run all configured models through the benchmark questions
3. Generate statistics and performance charts
4. Save results to `./results/[DATE]-[BENCHMARK-NAME]/`

## Configuration

Benchmark configurations are stored as YAML files in `configs/benchmarks/`. Each configuration file defines:

### Example Configuration

```yaml
# Benchmark name
name: "Multi-Benchmark"

# Dataset configuration
dataset:
  type: "multiple_choice"
  path: "benchmarks/easy_problems/linguistic_benchmark_multi_choice.json"

# Models to benchmark
models:
  gemma3-27-base:
    model: "gemma3:27b"
    service: "ollama"
  
  gemma3-27-simple-reflect:
    model: "gemma3:27b"
    service: "ollama_flow"
    flowchart: "simple_reflect"

# Execution parameters
execution:
  rounds: 10           # Number of rounds per model
  num_retries: 3       # Retries for failed questions
  temperature: 0       # Model temperature (0 = deterministic)
  max_tokens: 2048     # Max tokens per response

# Output configuration
output:
  base_dir: "./results"
  save_answers: true
  save_stats: true
  create_charts: true
```

### Configuration Sections

#### Dataset Configuration

The `dataset` section specifies which benchmark dataset to use:

- **type**: Dataset type (`"multiple_choice"` or `"free_form"`)
- **path**: Path to the dataset JSON file (relative to project root)

#### Model Configuration

The `models` section defines which models to benchmark. Each model has:

- **model**: The base model name (e.g., `"gemma3:27b"`, `"llama3.3"`)
- **service**: How to invoke the model:
  - `"ollama"`: Direct Ollama API
  - `"ollama_flow"`: Use a flowchart workflow (requires `flowchart` parameter)
  - `"ollama_reflector"`: Use reflection pattern
  - `"dummy"`: Testing service (always returns "A")
- **flowchart**: (Optional) Name of flowchart to use with `ollama_flow` service

#### Execution Configuration

The `execution` section controls how the benchmark runs:

- **rounds**: Number of times to run each model through the entire dataset
- **num_retries**: How many times to retry a question if the model returns invalid output
- **temperature**: Sampling temperature for models (0.0 to 1.0)
- **max_tokens**: Maximum tokens per model response

#### Output Configuration

The `output` section controls result storage:

- **base_dir**: Base directory for all benchmark results
- **save_answers**: Whether to save individual model answers
- **save_stats**: Whether to save aggregate statistics
- **create_charts**: Whether to generate performance charts

## CLI Usage

### Running Benchmarks

```bash
# Run with default config
pithos-benchmark

# Run with custom config
pithos-benchmark --config my_benchmark.yaml

# Run specific models only
pithos-benchmark --models gemma3-27-base gemma3-27-simple-reflect

# Run with custom number of rounds
pithos-benchmark --rounds 5

# Run with custom output directory
pithos-benchmark --output-dir ./my_results

# Skip chart generation
pithos-benchmark --no-charts

# Override temperature
pithos-benchmark --temperature 0.3
```

### Generating Reports

```bash
# Generate report from most recent results
pithos-benchmark report

# Generate report from specific results directory
pithos-benchmark report --path ./results/2024-03-12-Multi-Benchmark

# Generate report from specific date
pithos-benchmark report --date 2024-03-12
```

### Listing Resources

```bash
# List all available benchmark configs
pithos-benchmark list-configs

# List all available datasets
pithos-benchmark list-datasets
```

## Dataset Management

### Dataset Structure

Benchmarks use structured JSON datasets. pithos supports multiple dataset types:

#### Multiple Choice Dataset

```json
[
  {
    "question": "What is the capital of France?",
    "correct_answer": "Paris",
    "multiple_choice": [
      "London",
      "Paris",
      "Berlin",
      "Madrid"
    ]
  },
  {
    "question": "Which element has atomic number 1?",
    "correct_answer": "Hydrogen",
    "multiple_choice": [
      "Helium",
      "Hydrogen",
      "Oxygen",
      "Carbon"
    ]
  }
]
```

#### Free-Form Dataset

```json
[
  {
    "question": "Explain the concept of recursion in programming.",
    "correct_answer": "Recursion is when a function calls itself...",
    "evaluation_criteria": "Answer should mention self-reference and base case"
  }
]
```

### Adding New Datasets

1. **Create your dataset JSON file** in the appropriate format
2. **Save it** to a location in the project (e.g., `benchmarks/easy_problems/my_dataset.json`)
3. **Create or modify a config file** to reference your dataset:

```yaml
dataset:
  type: "multiple_choice"  # or "free_form"
  path: "benchmarks/easy_problems/my_dataset.json"
```

4. **Run the benchmark:**

```bash
pithos-benchmark --config configs/benchmarks/my_config.yaml
```

Or override the dataset directly:

```bash
pithos-benchmark --dataset benchmarks/easy_problems/my_dataset.json
```

### Dataset Abstraction API

You can also programmatically work with datasets:

```python
from benchmarks.easy_problems.dataset import load_dataset

# Load a multiple choice dataset
dataset = load_dataset("multiple_choice", "path/to/dataset.json")

# Access questions
questions = dataset.get_questions()
first_question = dataset[0]

# Export to JSON
dataset.to_json("output.json")
```

## Creating Custom Benchmarks

### Step 1: Create Dataset

Create a JSON file with your questions:

```json
[
  {
    "question": "Your question here",
    "correct_answer": "The correct answer",
    "multiple_choice": ["Option A", "Option B", "Option C", "Option D"]
  }
]
```

### Step 2: Create Configuration

Create a YAML config file in `configs/benchmarks/`:

```yaml
name: "My Custom Benchmark"

dataset:
  type: "multiple_choice"
  path: "benchmarks/my_dataset.json"

models:
  my-model:
    model: "gemma3:27b"
    service: "ollama"

execution:
  rounds: 5
  num_retries: 3
  temperature: 0
  max_tokens: 2048

output:
  base_dir: "./results"
  save_answers: true
  save_stats: true
  create_charts: true
```

### Step 3: Run Your Benchmark

```bash
pithos-benchmark --config configs/benchmarks/my_custom_benchmark.yaml
```

## Understanding Results

### Output Structure

After running a benchmark, results are saved to:

```
./results/[DATE]-[BENCHMARK-NAME]/
├── llm_outputs/
│   ├── round_1/
│   │   ├── final_answers-model1.json
│   │   └── final_answers-model2.json
│   ├── round_2/
│   └── ...
└── tables_and_charts/
    └── performance_chart.png
```

### Answer Files

Each `final_answers-[model].json` contains:

- **question**: The original question text
- **correct_letter**: The correct answer (A, B, C, or D)
- **model_answer**: The full model response
- **json_answer**: Extracted JSON answer
- **json_answer_letter**: The letter chosen by the model
- **score**: 100 if correct, 0 if incorrect
- **invalid_answer_letter**: 1 if model gave invalid output, 0 otherwise

### Performance Charts

The `performance_chart.png` shows:

- Accuracy percentage for each model
- Comparison across all tested models
- Visual indication of relative performance

### Statistics

Statistics include:

- **Mean score**: Average accuracy across all rounds
- **Standard deviation**: Consistency of performance
- **Invalid outputs**: Number of malformed responses
- **Per-question breakdown**: Success rate for each question

## Advanced Topics

### Model Services

Different service types offer different capabilities:

- **ollama**: Direct model calls, fastest, no special patterns
- **ollama_flow**: Executes flowchart workflows, enables complex reasoning patterns
- **ollama_reflector**: Built-in reflection for self-correction
- **ollama_stepper**: Adds "step by step" reasoning prompt

### Flowchart Integration

When using `ollama_flow`, specify a flowchart from `configs/flowcharts/`:

```yaml
models:
  gemma-refined-reflect:
    model: "gemma3:27b"
    service: "ollama_flow"
    flowchart: "refined_explicit_reflect"
```

This runs the model through the specified flowchart for each question.

### Parallel Execution

Currently, benchmarks run sequentially. Future versions may support:

- Parallel model execution
- Distributed benchmark runs
- Real-time progress tracking

## Troubleshooting

### Model Not Found

**Error:** `Model 'xyz' not found in config`

**Solution:** Ensure the model name matches exactly in your config, or use `--models` to specify only existing models.

### Dataset Not Found

**Error:** `Dataset file not found: ...`

**Solution:** Check that the `dataset.path` in your config is correct and the file exists.

### Invalid Model Response

If models frequently produce invalid responses:

1. Increase `execution.num_retries`
2. Check model compatibility with your prompt format
3. Verify the model is properly installed in Ollama

### Missing Dependencies

Ensure all required packages are installed:

```bash
pip install -e .
```

## Best Practices

1. **Start Small**: Test with 1-2 rounds before running full benchmarks
2. **Version Datasets**: Keep dataset files in version control
3. **Document Configs**: Add comments to YAML configs explaining choices
4. **Regular Baselines**: Establish baseline scores for comparison
5. **Track Results**: Keep historical results for trend analysis

## Example Workflows

### Compare Reflection Strategies

```yaml
models:
  base:
    model: "gemma3:27b"
    service: "ollama"
  
  simple-reflect:
    model: "gemma3:27b"
    service: "ollama_flow"
    flowchart: "simple_reflect"
  
  refined-reflect:
    model: "gemma3:27b"
    service: "ollama_flow"
    flowchart: "refined_explicit_reflect"
```

### Test Multiple Models

```yaml
models:
  gemma-27b:
    model: "gemma3:27b"
    service: "ollama"
  
  llama-33b:
    model: "llama3.3"
    service: "ollama"
  
  phi-4:
    model: "Phi4"
    service: "ollama"
```

### Quick Validation

```bash
pithos-benchmark --models my-model --rounds 1 --no-charts
```

## Future Enhancements

Planned features:

- [ ] Support for additional dataset types (coding challenges, reasoning tasks)
- [ ] Real-time benchmark monitoring dashboard
- [ ] Automated regression testing
- [ ] Cost tracking for API-based models
- [ ] Confidence intervals and statistical significance testing
- [ ] Benchmark dataset versioning and provenance
