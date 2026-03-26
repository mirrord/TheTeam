# pithos

pithos is an agentic LLM interaction framework. With it, you can spin up new models quickly, switch rapidly back and forth between conversations (contexts), and direct your agents to follow complex flowcharts.


## Installation
First, install your preferred version of torch.
[[instructions & links TBD]]
If you want to visualize flowcharts for agents, you need to install pygraphviz as well:
(side note: this is annoying, can I remove it later in favor of another lib?)
### WINDOWS:
1. install from https://graphviz.org/download/
2. run this command (you may need to change the file locations to match your system): ```python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz```
3. ```pip install graphviz```
4. add the graphviz binary location to your PATH env var


then clone this repo and execute:
```pip install .```

### LINUX:
clone this repo and:
`pip install .`

What about graphviz, you ask? I don't know, I didn't try to install it there

## Usage
Demos:
`$> pithos-demo`
(make sure you have pygraphviz installed)

Agents:
```
# Create a new agent
agent = OllamaAgent("glm-4.7-flash")

# Prompt the model once & show its response
rsp = agent.send("Why is the sky blue? Be concise.")
print(rsp)
```

Contexts:
```
# Create a new context (conversation)
agent.create_context("session1")

agent.send("Why is the sky blue? Be concise.")
agent.send("Wow! How cool is that??")
# In the context "session1" we establish a reply style
print(
    agent.send(
        "What is the capital of France? Be concise. Answer as if you're a pirate."
    )
)

# Create a new context and switch to it immediately
agent.switch_context("session2")

# instructions & messages from other contexts do not carry over - each new context is independent 
print(
    agent.send(
        "Why is the sky blue? Be concise. If you have been given a previous style of reply, use it. If not, say so."
    )
)
agent.send("Wow! How cool is that??")
```

## Benchmarking

The command `pithos-benchmark` will execute the benchmarking tool, which current just runs the easy-questions-llms-get-wrong benchmark against configured agents. Configure this tool via the config.py under benchmarks/easy_problems (yuck!)

