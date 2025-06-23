<!-- # counter-argument-generation -->

# Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.05328-b31b1b.svg)](https://arxiv.org/abs/2503.05328)
<!-- [![GitHub](https://img.shields.io/badge/GitHub-anaryegen%2Fcounter--argument--generation-blue)](https://github.com/anaryegen/counter-argument-generation) -->

## 📋 Abstract

This work investigates the role of dynamic external knowledge integration in improving counter-argument generation using Large Language Models (LLMs). While LLMs show promise in argumentative tasks, their tendency to generate lengthy, potentially unfactual responses highlights the need for more controlled and evidence-based approaches. We introduce a new manually curated dataset of argument and counter-argument pairs specifically designed to balance argumentative complexity with evaluative feasibility. We also propose a new LLM-as-a-Judge evaluation methodology that shows a stronger correlation with human judgments compared to traditional reference-based metrics. Our experimental results demonstrate that integrating dynamic external knowledge from the web significantly improves the quality of generated counter-arguments, particularly in terms of relatedness, persuasiveness, and factuality.

## 📊 Dataset

### CANDELA-Based Dataset
- **Source**: Built upon the CANDELA corpus from r/ChangeMyView subreddit debates
- **Size**: 150 high-quality argument-counter-argument pairs  
- **Format**: 3-sentence structured arguments focusing on main claim, supporting evidence, and examples
- **Data**: [🤗 Counter-argument](https://huggingface.co/datasets/HiTZ/counter-argument)

### Data Statistics
| Component | Original | Intermediate | Final |
|-----------|----------|--------------|-------|
| **Arguments** | 16 sentences (372 words) | 3 sentences (83 words) | 3 sentences (61 words) |
| **Counter-arguments** | 30 sentences (921 words) | 5 sentences (165 words) | 3 sentences (72 words) |

## 📈 Results

### Performance Rankings
| Model | Human Eval Rank | LLM Judge Rank | Key Strengths |
|-------|----------------|----------------|---------------|
| Command R+ + External | 1 | 1 | Opposition, Factuality |
| Mistral-7B + External | 2 | =3 | Persuasiveness, Relatedness |
| Command R+ | 3 | 2 | Strong parametric knowledge |
| Mistral-7B | 4 | =3 | Baseline performance |

### Evaluation Dimensions
The study evaluated counter-arguments across five key dimensions:
1. **Opposition**: How well the counter-argument opposes the original claim
2. **Relatedness**: Relevance to the original argument  
3. **Specificity**: Level of detail and precision
4. **Factuality**: Accuracy of presented information
5. **Persuasiveness**: Convincing power of the argument

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{yeginbergen2025dynamic,
  title={Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models},
  author={Yeginbergen, Anar and Oronoz, Maite and Agerri, Rodrigo},
  booktitle={ACL Findings},
  year={2025}
}
