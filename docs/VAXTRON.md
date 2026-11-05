# Running `paper-extracting` on Vaxtron

This guide describes how to run the cohort metadata extraction pipeline on the **UMCG Vaxtron HPC cluster** with an existing configured runtime. For the earlier Nibbler workflow, see the [main README](../readme.md).

## Prerequisites

- Authorized Vaxtron access, project storage and permission to submit GPU jobs.
- A repository checkout containing `run_vaxtron_cohort.sbatch` and `src/run_vaxtron_cohort.sh`.
- A prepared `.venv.cluster/` and `.runtime/` with the compiled `llama-server`, the Gemma GGUF model, and the GLM-OCR model and projector.
- A scientific PDF that you are permitted to process.

The steps below use a **prepared runtime**. See [Runtime setup](#runtime-setup) if you are starting from a new checkout.

## Run an extraction

### 1. Connect and open the repository

On your **local computer**, use the SSH alias configured for your Vaxtron account:

```bash
ssh YOUR_CONFIGURED_VAXTRON_ALIAS
```

On **Vaxtron**, navigate to your repository checkout:

```bash
cd /path/to/paper-extracting
pwd
git rev-parse --short HEAD
```

Replace the example alias and path with your own settings.

### 2. Check the runtime

From the repository root, verify that the necessary files are present:

```bash
for file in \
  .venv.cluster/bin/python \
  .runtime/llama.cpp/build/bin/llama-server \
  .runtime/GGUF/gemma-4-31B-it-Q4_K_M.gguf \
  .runtime/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf \
  .runtime/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf \
  config.cohort.toml \
  prompts/prompts_cohort.toml \
  run_vaxtron_cohort.sbatch \
  src/run_vaxtron_cohort.sh; do
  if [ -s "$file" ]; then
    printf 'OK: %s\n' "$file"
  else
    printf 'MISSING: %s\n' "$file"
  fi
done
```

The Python and `llama-server` binaries must also be executable. Resolve any missing prerequisites before submitting a job.

### 3. Choose the PDF and output filename

On **Vaxtron**, select the input paper and a distinct output filename:

```bash
PDF="data/oncolifes.pdf"
OUTPUT="oncolifes_$(date -u +%Y%m%dT%H%M%SZ).xlsx"
ls -lh "$PDF"
printf 'Output: %s\n' "$OUTPUT"
```

Replace the example PDF with your own authorized input. Confirm that `ls` finds the file before continuing.

### 4. Submit the Slurm job

From the repository root, using the same terminal session:

```bash
SYNC_OUTPUT_ENABLE=0 COHORT_PROMPT_SCHEMA_PUSH_GIT=0 \
  sbatch --export=ALL run_vaxtron_cohort.sbatch \
  -p all --pdfs "$PDF" -o "$OUTPUT"
```

Record the job ID printed by Slurm. Here, `-p all` is an **argument to the extraction program**, not a Slurm partition flag. Resource requests are defined in `run_vaxtron_cohort.sbatch` and may need to be adapted to your approved allocation.

The command disables automatic output transfer and prompt-schema Git pushes for this job. Always specify the intended input and output: the batch script otherwise starts a default extraction.

### 5. Monitor the job

On **Vaxtron**, substitute your actual Slurm job ID:

```bash
JOB_ID=123456
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
```

Inspect the wrapper logs in the submission directory:

```bash
tail -n 80 "cohort_extract-${JOB_ID}.out"
tail -n 80 "cohort_extract-${JOB_ID}.err"
```

Detailed model, OCR and extraction logs are normally saved in `logs/runs/<job_id>/`, unless `RUN_ID` or `RUN_DIR` was overridden. A job can disappear from `squeue` once it finishes; use `sacct` to check its final state.

### 6. Inspect the output

On **Vaxtron**, locate the workbook (using the output name from Step 3):

```bash
ls -lh "$OUTPUT"
```

Review the generated metadata against the source publication and the relevant MOLGENIS schema, including participant counts, ontology terms and references. A successful Slurm job does not establish scientific correctness.

### 7. Retrieve the workbook

On your **local computer**, transfer the reviewed file using your configured SSH alias:

```bash
rsync -avhP \
  YOUR_CONFIGURED_VAXTRON_ALIAS:/path/to/paper-extracting/OUTPUT_FILE.xlsx \
  ./
```

Replace the remote path and filename with your actual output. Transfer and store only materials you are authorized to use outside the cluster.

## Pipeline and configuration

The Vaxtron runner checks the input PDFs and uses **GLM-OCR** when ordinary PDF text extraction is insufficient. In prefetch mode, it saves OCR text and stops the OCR server before launching two local **Gemma 4 31B** inference servers. It then prepares the load balancer and metadata configuration and runs `src/main_cohort.py` to produce a MOLGENIS-oriented Excel workbook.

| Component | Configuration in the inspected scripts |
| --- | --- |
| Python module | `Python/3.10.4-GCCcore-11.3.0` |
| CUDA module | `CUDA/13.3.1` |
| llama.cpp | Commit `e21cdc11a0461d8b0cbd28cc356d993bf6be7282` |
| Main model | `gemma-4-31B-it-Q4_K_M.gguf` |
| Gemma revision | `31311383e7f05e21e39dc69b39285be9cdeb1257` |
| OCR | `GLM-OCR-Q8_0.gguf` and matching `mmproj` |
| OCR revision | `65a42de1148dbed2297e922b5dbc7d9b70c36578` |
| Python dependencies | `requirements.lock` |
| Slurm resources | 1 node, 2 GPUs (`gpu:rtx6000:2`), 4 CPUs, 64 GB RAM, 1 hour |
| Metadata profile | `UMCGCohortsStaging` |
| EMX2 source default | `molgenis/molgenis-emx2` at ref `main` |

These are **script settings**, not a guarantee of current cluster availability. The EMX2 profile selects metadata tables and columns; it is not a database connection. Because the default EMX2 ref `main` can change, preserve the actual schema revision and snapshot when comparing research results.

## Runtime setup

The extraction steps assume a working `.runtime/` and `.venv.cluster/`. The Vaxtron setup scripts can clone or update `llama.cpp`, build CUDA support, download large pinned models and install Python dependencies. They can also change existing runtime files. Review the scripts and your institution's HPC guidance before using them to prepare a new environment.

In particular, `setup_vaxtron_models.sbatch` expects `llama-server` to have been built already; it is **not** a complete fresh-install procedure by itself.

## Run records and troubleshooting

For scientific reproducibility, retain the project commit, input paper/version, model and EMX2 schema revisions, effective configuration and prompts, job logs and human validation notes. Keep inputs and outputs in approved storage.

If a run fails, check the Slurm `.out`/`.err` files and `logs/runs/<job_id>/logs/`. If metadata looks incorrect or incomplete, check the original PDF, saved OCR text and schema inputs.

Do not commit private PDFs, model weights, virtual environments, run outputs, logs, tokens or credentials to this public repository. The inference services are intended for HPC job use, not public network access.
