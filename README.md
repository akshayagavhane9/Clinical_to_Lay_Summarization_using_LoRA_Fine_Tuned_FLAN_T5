Clinical-to-Lay Summarization using LoRA Fine-Tuned FLAN-T5

This repository contains the complete code, dataset workflow, and trained adapters for fine-tuning the FLAN-T5 model using LoRA (Low-Rank Adaptation) to convert biomedical abstracts into easy-to-read patient-friendly summaries, leveraging the BioLaySumm 2025 (PLOS) dataset.

🚀 Quick Setup (macOS / Colab / VS Code)
Option A — Google Colab (fastest)

Open the notebook Clinical_to_Lay_Summarization.ipynb in Colab.

Run the steps sequentially:

Step 1 → Environment setup

Step 2 → Dataset loading

Step 3–8 → Training → Evaluation → Error Analysis → Inference demo

Log in to Hugging Face (if prompted):

from huggingface_hub import login
login()

Option B — Local macOS (VS Code + Python venv)
# Clone repository
git clone https://github.com/akshayagavhane9/Clinical_to_Lay_Summarization_using_LoRA_Fine_Tuned_FLAN_T5.git
cd Clinical_to_Lay_Summarization_using_LoRA_Fine_Tuned_FLAN_T5

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch Jupyter or VS Code and open the main notebook

Option C — Conda Environment (optional)
conda env create -f environment.yml
conda activate clinical-lay-lora

🧩 Requirements

All tested versions used in Colab/macOS 12:

transformers==4.41.2
datasets==2.20.0
accelerate==0.30.0
peft==0.11.1
torch>=2.1
rouge-score==0.1.2
evaluate==0.4.1
nltk==3.8.1
sentencepiece==0.1.99
pandas
matplotlib
tqdm
scikit-learn

🧠 Project Workflow
Step	Component	Purpose
1	Environment Setup	Install dependencies, check GPU
2	Dataset Loading	Load BioLaySumm 2025 (PLOS)
3	Baseline Evaluation	Test FLAN-T5 zero-shot performance
4	LoRA Fine-Tuning	Train model with non-zero loss
5	Hyperparameter Optimization	Test ≥ 3 configs + select best
6	Final Evaluation	Compare baseline vs fine-tuned
7	Error Analysis	Identify weakest samples & biases
8	Inference Demo	Generate lay summaries + latency test
📊 Output Structure
.
├── notebooks/
│   └── Clinical_to_Lay_Summarization.ipynb   # main notebook
├── runs/
│   ├── sweep/r8_a32_lr2e-4/                  # best LoRA checkpoint
│   └── flan-t5-small-lora-manual/
├── outputs/
│   ├── hpo_results.csv
│   ├── final_eval_metrics.csv
│   ├── error_analysis.csv
│   ├── baseline_val_preds.jsonl
│   └── final_val_preds.jsonl
├── requirements.txt
├── environment.yml
└── README.md

🧪 Reproducibility

Best HPO config: r8_a32_lr2e-4

Validation ROUGE-Lsum: ↑ from 0.08 → 0.22 (+168%)

Flesch Readability: slightly lower (more precise medical phrasing)

Fine-Tuned Model: Hugging Face Hub → akshayagavhane999/flan-t5-small-clinical-translator-lora-timestamp

📈 Key Results
Metric	Baseline	Fine-Tuned	Δ (Improvement)
ROUGE-1	0.11	0.34	+209 %
ROUGE-2	0.04	0.11	+175 %
ROUGE-Lsum	0.09	0.20 – 0.22	+150 %
Flesch	27.3	20.9	-24 % (more medically precise)
% Long Words	0.067	0.060	-10 % (simpler phrasing)
🩺 Inference Demo (Truncate vs Chunk)

Example input:
“Lymphatic filariasis is a mosquito-borne disease caused by filarial worms…”

Mode	Summary	Latency (s)
Truncate	Lymphatic filariasis is a mosquito-borne disease caused by filarial worms …	2.3 s
Chunk	Same content processed across 8 chunks → smoother coverage	10.2 s
🧩 Error Analysis Highlights

Model underperforms on ultra-long abstracts (> 10k tokens).

Some redundant phrasing appears in generated lay text.

Future work: implement chunk-aware summarization and RLHF refinement for stylistic control.

🔄 Future Improvements

Integrate adapter-merging for multi-domain adaptation

Use text stat feedback loops (Flesch, jargon count) for dynamic loss

Deploy via Gradio UI for patient-facing demo

⚙️ Hugging Face Deployment

You already logged in in Colab:

model.push_to_hub("akshayagavhane999/flan-t5-small-clinical-translator-lora")
tokenizer.push_to_hub("akshayagavhane999/flan-t5-small-clinical-translator-lora")

🧾 Citation

If you use this work, please cite:

@project{akshayagavhane2025_clinical_lora,
  title   = {Clinical-to-Lay Summarization using LoRA Fine-Tuned FLAN-T5},
  author  = {Akshaya Gavhane, Ritwik Giri et al.},
  year    = {2025},
  note    = {Northeastern University MSIS Course Project}
}

💡 Author & Acknowledgements

Developed by Akshaya Gavhane with guidance from Ritwik Giri as part of Northeastern University’s Prompt Engineering & AI Fine-Tuning Project (Fall 2025).
Dataset: BioLaySumm 2025 (PLOS), via Hugging Face Hub.
