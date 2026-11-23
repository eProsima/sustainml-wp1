# SustainML – Work Package 1 (WP1)

[![SustainML](.figures/SustainML_GitHub.png)](https://sustainml.eu/)

<br>

<div class="menu" align="center">
    <strong>
        <a href="https://eprosima.com/index.php/downloads-all">Download</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.readthedocs.io/en/latest">Docs</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://sustainml.eu/showroom/news">News</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="https://www.linkedin.com/company/eprosima/">LinkedIn</a>
        <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
        <a href="mailto:info@eprosima.com">Contact Us</a>
    </strong>
</div>

<br><br>

<div class="badges" align="center">
    <a href="https://github.com/eProsima/sustainml-wp1/releases"><img alt="Releases" src="https://img.shields.io/github/v/release/eProsima/sustainml-wp1?sort=semver"/></a>
    <a href="https://github.com/eProsima/sustainml-wp1/issues"><img alt="Issues" src="https://img.shields.io/github/issues/eProsima/sustainml-wp1.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp1/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/eProsima/sustainml-wp1.svg"/></a>
    <a href="https://github.com/eProsima/sustainml-wp1/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/eProsima/sustainml-wp1.svg"/></a>
    <a href="https://SustainML.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://img.shields.io/readthedocs/sustainml/latest"/></a>
</div>

<br><br>

This repository contains components developed as part of **Work Package 1 (WP1)** of the SustainML project.

WP1 focuses on **understanding the machine learning problem** described by the user and **proposing a model** for it.

## Scope

This repository contains three nodes of the SustainML Framework:

- **App Requirements** (`app_requirements_node.py`): a placeholder in the current version, which publishes the same
  fixed list of application requirements for every task.
- **ML Model Metadata** (`ml_model_metadata_node.py`): finds the ML goal of the problem (with a local `llama3` model
  through Ollama) and describes uploaded datasets (`AutoDDG`).
- **ML Model Provider** (`ml_model_provider_node.py`): proposes a machine learning model for the goal, from the Neo4j
  knowledge graph of models and a semantic search over it (`rag`), the YOLO models of the Axelera board or the U-Net
  models of the FPGA, or takes the model chosen by the user. It also starts the Hugging Face MCP server
  (`hf_mcp_server.py`) used to search Hugging Face models.

## Usage

The WP1 nodes are not intended to operate independently: they are integrated into the SustainML pipeline as part of
the `sustainml_modules` package of the [SustainML Library](https://github.com/eProsima/SustainML-Library), where this
repository is a Git submodule.

For installation, configuration, and execution instructions, refer to the main framework repository:

https://github.com/eProsima/SustainML-Framework
