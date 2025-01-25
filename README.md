# LLM Load Testing with Enron Emails

This repository provides a framework for **realistic load tests** of Large Language Models (LLMs) using **randomly selected emails** from the **Enron dataset**. By simulating real-world communications, we achieve more accurate performance metrics and can better evaluate LLM behavior at scale.

## Table of Contents
1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [VS Code Debug Config](#vs-code-debug-config)  
5. [Why Realistic Load Tests?](#why-realistic-load-tests)  
6. [Comparisons to Other Approaches](#comparisons-to-other-approaches)  
7. [Results in the Dashboard](#results-in-the-dashboard)

---

## Overview

The **Enron Email dataset** is a large collection of real-world corporate emails. We harness it by:
- Randomly selecting one email body per test request,
- Generating a random or semi-random question for the LLM about that email,
- Sending these queries to the LLM, thus creating a stream of requests that mirror practical usage patterns.

By using **true corporate communication** samples, we capture complexities such as technical terms, personal notes, and compliance topics—leading to **realistic token lengths** for both **input** (the email) and **output** (the LLM’s response). This is far more authentic than what you’d get from synthetic or purely encyclopedic data.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/miboettc/llm-load-jukebox.git
   cd llm-load-jukebox
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Enron Dataset from Kaggle**:  
   The Enron dataset is downloaded from [Kaggle](https://www.kaggle.com/). You need:
   - A **Kaggle account** (sign up on the website).  
   - A **Kaggle API token** to authenticate your local environment.  

   Typically, you can do the following:
   1. Go to your Kaggle account settings and select **Create New API Token** (this downloads `kaggle.json`).  
   2. Place `kaggle.json` under `~/.kaggle/` (Linux/macOS) or `C:\Users\<YourUser>\.kaggle\` (Windows).  
   3. Ensure that file has correct permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).

   Once set up, the scripts in this repo can automatically authenticate and download the Enron dataset.

---

## Usage

Below are commands matching the Locust debug configurations in `launch.json`.

### Headless Mode

```bash
locust --headless -u 10 -r 1 --run-time 5m -f llm_load_jukebox/enron_locust.py
```

Explanation:

```
--headless: Runs Locust without the web interface.
-u 10: Spawns 10 concurrent users.
-r 1: Ramp up at 1 user/second.
--run-time 5m: Stop after 5 minutes.
-f: Points to our main Locust file (enron_locust.py).
```

### Web UI Mode

```bash
locust -f llm_load_jukebox/enron_locust.py
```

Open [http://localhost:8089](http://localhost:8089) in your browser to view the Locust web interface.

---

## VS Code Debug Config

For easy debugging, we provide a [**launch.json**](.vscode/launch.json) file. It defines:
- **Locust Headless** mode, mirroring the `locust --headless ...` command  
- **Locust Web UI** mode  
- **Python: enron_llm_processor.py** for debugging local scripts  


---

## Why Realistic Load Tests?

Load tests on **real data** (like the Enron emails) are essential because:

- Synthetic or trivial prompts often fail to represent genuine user traffic and corporate-lingo complexities.
- They show how an LLM handles domain-specific jargon, personal references, and compliance issues—key factors for production readiness.
- **Crucially**, they lead to more realistic **token lengths** for prompts and responses, mirroring actual usage scenarios.
- By running tests at scale, we discover bottlenecks, latencies, and memory usage patterns that remain hidden with synthetic or simplistic text.

In short, **realistic data** provides a **relevant** stress test, ensuring your LLM solution can handle production-like workloads.

---

## Comparisons to Other Approaches

- **Synthetic Prompt Generators**  
  - Pros: Easy to customize, no dataset licensing.  
  - Cons: Lacks authenticity; typical token lengths for real emails are absent, so results can be misleading.

- **Public Domain Datasets (e.g., Wikipedia)**  
  - Pros: Large, easily accessible.  
  - Cons: Encyclopedic style can differ wildly from corporate or personal communication. Often yields very different token usage patterns.

- **Other Email/Chat Datasets**  
  - Pros: Possibly domain-specific.  
  - Cons: Might be smaller or less widely recognized; Enron remains a canonical reference that’s well-studied and widely tested.

Hence, the **Enron** dataset stands out for **realism** (corporate language and complexity), **size**, and **broad acceptance** in research—providing real-world token distributions for input and output.

---

## Results in the Dashboard

When you run tests, Locust’s **dashboard** can display **two custom lines**:  
1. **Time to First Token**:  
   - Listed as a separate request/endpoint name in the results table.  
   - The **request size** column reflects the **number of input tokens** (from the selected Enron email).  
2. **End-to-End Latency**:  
   - Listed as another line in the same or separate results table.  
   - The **request size** column here indicates the **number of output tokens** returned by the LLM.

By splitting these metrics, you can easily see how quickly the LLM started streaming back tokens (TTFT) vs. how long it took to complete the entire response (E2E). The corresponding “size” columns help correlate **latency** with **token volume**—giving deeper insight into your model’s scaling behavior.

---


## Limitation

- Currently only works with ollama deployments, but the code can easily be extended to performance test other API's 
