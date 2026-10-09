# MASTER PROMPT: Build a Production-Quality Restaurant Order Demand Prediction System

## 1. Role and Mission

Act as a senior Machine Learning Engineer, Senior Python Engineer, MLOps Engineer, and Software Architect.

Build a complete, professionally engineered, portfolio-ready machine learning project titled:

**Restaurant Order Demand Prediction**

Technology stack:
- Python 3.11 or another explicitly justified, compatible Python version.
- pandas and NumPy for data processing.
- scikit-learn for preprocessing, modeling, and evaluation.
- Matplotlib and Seaborn for exploratory analysis and visualizations.
- pytest for automated testing.
- Ruff for linting and formatting.
- Joblib for model persistence, where appropriate.
- PyYAML or another lightweight configuration mechanism if justified.
- Streamlit or FastAPI for a user-facing prediction interface, with a clear justification for the choice.
- GitHub Actions for automated testing and quality checks.

Choose stable, mutually compatible dependency versions. Pin dependencies or define sensible version constraints and provide a reproducible installation workflow.

Your responsibility is to implement the entire project, not merely provide instructions, snippets, a design document, or a notebook.

Inspect the available development environment and existing repository before making changes. If a repository already exists, preserve useful work and integrate with it instead of blindly overwriting files.

Work autonomously wherever reasonable. Do not stop after creating a plan. Implement the complete solution, run the available tests, resolve failures, and produce the final documentation.

If you have access to terminal execution, file editing, and test execution tools, use them. If external services or data sources are inaccessible, document the limitation and implement a reliable alternative rather than fabricating successful results.

## 2. Project Objective

Develop an end-to-end supervised machine learning system that predicts restaurant order demand from historical data.

The system should learn relationships between relevant features and a clearly defined demand target. Depending on the dataset, the target may be:
- Number of orders during a specified time interval.
- Number of orders for a restaurant during a specified time interval.
- A demand-related quantity explicitly defined in the dataset documentation.

Do not assume these targets are interchangeable.

Before implementing the pipeline, identify the dataset's actual prediction target, feature definitions, granularity, units, and temporal characteristics.

The project must answer the following questions:

1. What quantity is being predicted?
2. At what level is each prediction made?
3. What historical information is available at prediction time?
4. Which features influence the predictions?
5. How accurately does the system predict unseen demand?
6. How does the trained model compare with a simple baseline?
7. Which observations produce the largest errors?
8. How can a user obtain predictions using new input data?
9. How can another developer install, reproduce, test, and extend the system?

The final product must be a complete software repository with documented, executable workflows.

## 3. Dataset Selection and Data Integrity

Identify a real, publicly accessible dataset suitable for restaurant order-demand prediction.

Investigate appropriate datasets, including restaurant demand forecasting datasets available through Kaggle, UCI, GitHub-hosted public repositories, or other legitimate sources.

Prefer a dataset containing actual historical order counts or order volumes. A dataset involving food delivery demand, restaurant orders, or restaurant-level time-series demand is acceptable if its target and limitations are clearly explained.

Evaluate candidate datasets based on:
- Relevance to the business problem.
- Target-variable availability.
- Sample size and feature richness.
- Data quality.
- Time-related information.
- Availability of training and evaluation observations.
- Licensing and redistribution restrictions.
- Ease of reproducible access.

Do not choose a dataset solely because it is convenient to download.

### Dataset acquisition requirements

Implement a reliable data acquisition workflow.

Where practical:
- Provide a documented download or ingestion script.
- Support a user-specified local dataset path.
- Validate that expected files and columns exist.
- Report file size, row count, column count, missing values, and duplicate counts.
- Handle encoding and malformed input errors explicitly.
- Keep raw data separate from processed data.
- Record dataset source, download instructions, and licensing information.
- Document any manual download steps or authentication requirements.
- Make the project usable without embedding personal API keys or credentials.

Never commit API keys, credentials, private data, or restricted datasets.

Do not fabricate a dataset, source URL, dataset statistics, model metrics, or experimental results.

If the original dataset cannot be downloaded automatically, provide a documented manual ingestion path. If a small synthetic fixture is needed for testing, label it clearly as synthetic and never present its results as evidence of model performance on real restaurant demand.

### Data schema documentation

Document the actual schema after inspecting the selected dataset.

For each field, record:
- Column name.
- Data type.
- Meaning.
- Units, where applicable.
- Whether the field is an input feature, target, identifier, or metadata.
- Whether it is available at prediction time.
- Missing-value behavior.
- Any preprocessing requirements.

Create an explicit distinction between identifiers and predictive features.

## 4. Define the Prediction Problem Correctly

Before modeling, establish a formal prediction contract.

Document:
- The target variable and its units.
- The observation unit.
- The prediction horizon, if applicable.
- The information cutoff for making predictions.
- The expected input schema.
- The output schema.
- The evaluation methodology.
- The limitations of the selected dataset.

For example, if the data contains one row per restaurant per hour, establish whether the model predicts orders for the current hour, a future hour, or another explicitly defined interval.

Do not describe ordinary random-row prediction as future demand forecasting when it does not simulate future prediction conditions.

If the dataset is temporal, investigate a chronological train-validation-test split. Where appropriate, use a rolling or expanding-window validation strategy.

If the dataset is not temporal, use a suitable held-out evaluation strategy and clearly describe its assumptions.

If repeated restaurants, customers, stores, or other entities appear across rows, assess whether entity leakage is possible and choose an appropriate split strategy.

All split decisions must be justified in the README.

## 5. Professional Repository Architecture

Use a modular, maintainable Python package rather than placing the entire implementation inside a single notebook.

Start with an architecture similar to the following, adapting it when the selected dataset or deployment interface justifies a different arrangement.

restaurant-order-demand-prediction/
- README.md
- LICENSE
- pyproject.toml
- .gitignore
- .env.example, only if environment configuration is genuinely required
- Makefile or equivalent documented developer commands
- requirements-lock.txt or another reproducible dependency lock artifact, if supported by the chosen dependency manager
- .github/
  - workflows/
    - ci.yml
- configs/
  - model.yaml
  - data.yaml
- data/
  - raw/
  - interim/
  - processed/
  - README.md
- notebooks/
  - 01_exploratory_data_analysis.ipynb
  - 02_model_development.ipynb
- reports/
  - figures/
  - model_evaluation.md
- models/
  - .gitkeep
- src/
  - restaurant_demand/
    - __init__.py
    - config.py
    - logging_config.py
    - exceptions.py
    - data/
      - __init__.py
      - ingestion.py
      - validation.py
      - preprocessing.py
    - features/
      - __init__.py
      - build_features.py
    - models/
      - __init__.py
      - baseline.py
      - train.py
      - evaluate.py
      - predict.py
      - registry.py, if justified
    - monitoring/
      - __init__.py
      - drift.py, if implemented
    - api/
      - __init__.py
      - schemas.py
      - app.py
    - cli.py
- scripts/
  - download_data.py, if applicable
  - run_pipeline.py
- tests/
  - conftest.py
  - unit/
  - integration/
  - test_data_validation.py
  - test_preprocessing.py
  - test_training.py
  - test_prediction.py
  - test_api.py, if applicable
- docker/
  - Dockerfile, if justified
- docs/
  - ARCHITECTURE.md
  - DATA_DICTIONARY.md
  - MODEL_CARD.md
  - EXPERIMENTS.md
  - INTERVIEW_GUIDE.md
- app.py, only if required by the selected interface.

This is a suggested structure, not a requirement to create empty modules just to imitate enterprise software.

Keep the architecture appropriately sized for a strong individual portfolio project. Avoid unnecessary microservices, excessive abstractions, multiple orchestration frameworks, or infrastructure that provides no meaningful benefit.

Explain any significant departures from this structure.

### Architecture principles

Implement clear separation of concerns:

1. Data ingestion.
2. Data validation.
3. Exploratory analysis.
4. Feature preparation.
5. Model training.
6. Model evaluation.
7. Model persistence.
8. Inference.
9. Interface or API.
10. Testing and documentation.

Use explicit function interfaces and meaningful type annotations.

Avoid circular imports, hidden global state, duplicated preprocessing logic, unexplained magic constants, and business logic embedded in user-interface code.

Use configuration files for meaningful adjustable settings rather than scattering constants throughout the implementation.

## 6. Production-Quality Python Standards

All implementation code must be professional, readable, maintainable, and deeply commented where explanation adds value.

### Coding requirements

- Follow PEP 8.
- Use type hints for public functions, methods, and important data structures.
- Add Google-style or NumPy-style docstrings to public functions, classes, and modules where useful.
- Use descriptive names.
- Keep functions focused on one responsibility.
- Prefer explicit behavior over clever shortcuts.
- Use pathlib for filesystem operations.
- Use Python's logging module instead of relying on print statements for application diagnostics.
- Create custom exceptions when they meaningfully improve error handling.
- Validate external inputs at system boundaries.
- Handle expected failures with informative error messages.
- Avoid bare except clauses and swallowed exceptions.
- Avoid hardcoded machine-specific paths.
- Use context managers where appropriate.
- Avoid mutable default arguments.
- Avoid unsafe deserialization and unnecessary pickle loading.
- Ensure random seeds are configured consistently where deterministic behavior is expected.
- Keep training-only operations separate from inference-time operations.
- Avoid embedding credentials or sensitive configuration in source code.
- Ensure modules can be imported without downloading data, training models, starting servers, or executing other expensive side effects.

### Deep-commenting requirements

Comments must teach a developer why the code works, not merely repeat what each line says.

For example, explain:
- Why a chronological split is required.
- Why an identifier is excluded from model features.
- Why preprocessing must be fitted only on training data.
- Why a baseline is necessary.
- Why a particular evaluation metric is appropriate.
- Why a specific error-handling decision is made.
- Why the inference schema rejects a given input.
- Why a saved artifact must match the preprocessing and feature schema used during training.

Add comments around non-obvious logic, important assumptions, leakage prevention, and design trade-offs.

Do not add meaningless comments to every line. The goal is unusually clear, deeply explained code without sacrificing readability.

**Do not use emojis anywhere in source code, comments, docstrings, filenames, command output intentionally generated by the application, or technical documentation.** Use professional plain-text messages.

## 7. Data Validation and Preprocessing

Implement a dedicated validation layer.

Check for:
- Missing required columns.
- Unexpected or duplicate column names.
- Invalid data types.
- Missing or invalid target values.
- Duplicate observations.
- Negative order counts where they are invalid.
- Invalid dates and timestamps.
- Impossible values based on documented domain rules.
- Infinite numeric values.
- Unexpected categorical values.
- Invalid identifiers.
- Empty input datasets.
- Inconsistent training and inference schemas.

Do not automatically delete every row containing missing values.

Choose and document missing-data policies based on the dataset and modeling objective.

Separate validation failures from recoverable data-quality warnings.

Where reasonable, produce a data-quality report containing:
- Total rows and columns.
- Missingness by field.
- Duplicate counts.
- Target distribution.
- Numeric feature summaries.
- Categorical cardinalities.
- Outlier summaries.
- Date range and temporal coverage, when applicable.

Outliers should be investigated rather than automatically removed. Genuine demand spikes may be business-critical observations.

Use scikit-learn Pipelines and ColumnTransformer when appropriate to ensure preprocessing is fitted only on training data and applied consistently during inference.

Persist the complete fitted preprocessing-and-model pipeline whenever practical.

Never fit a scaler, imputer, encoder, feature selector, or other learned transformation on the full dataset before evaluation.

Do not expose the target variable or any information derived from future observations as a predictor.

## 8. Exploratory Data Analysis

Create a reproducible EDA workflow in the notebook and, where appropriate, reusable reporting functions in the package.

Investigate:
- Dataset shape and schema.
- Data types.
- Missing values.
- Duplicate records.
- Target distribution.
- Descriptive statistics.
- Numeric feature distributions.
- Categorical feature frequencies.
- Feature-target relationships.
- Correlations, interpreted cautiously.
- Outliers and unusual observations.
- Time-of-day, day-of-week, seasonal, or holiday patterns if relevant fields exist.
- Restaurant-level differences if restaurant identifiers or categories exist.
- Demand trends over time if temporal information exists.

Generate clear, labeled visualizations with sensible titles, axes, legends, and units.

Every chart must answer a meaningful question.

Explain important findings in markdown cells instead of leaving charts unexplained.

Do not manufacture insights merely to make the project appear impressive.

If the data lacks a field required for a particular analysis, state that limitation instead of inventing the field.

Keep the notebook readable and avoid making it the only way to execute the project.

## 9. Feature Engineering

Create a dedicated feature-engineering layer.

Use only information available at the actual prediction cutoff.

Depending on the data, potentially useful features may include:
- Hour of day.
- Day of week.
- Month.
- Weekend indicator.
- Holiday indicator, only when supported by reliable data.
- Restaurant category.
- Restaurant location or region, where legitimate and available.
- Historical demand lags.
- Rolling historical demand statistics.
- Historical averages by time period.
- Other justified business-relevant variables.

These are candidates, not mandatory features. Implement only those supported by the dataset.

### Time-series feature safety

If lag or rolling features are appropriate:
- Sort observations by entity and timestamp.
- Shift historical demand so the current target cannot leak into its own predictors.
- Group by restaurant or another entity when necessary.
- Ensure rolling statistics use only past observations.
- Prevent future information from entering historical training rows.
- Avoid computing test-set features from unavailable future targets.
- Clearly define how historical features are obtained for new predictions.

A feature that cannot be generated at inference time must not silently enter the production pipeline.

Write unit tests for leakage-sensitive transformations.

## 10. Modeling Strategy

Build a meaningful comparison between a simple baseline and at least one stronger model.

### Required baseline

Choose a baseline that is appropriate for the target and data structure.

Possible baselines:
- Training-set mean predictor.
- Training-set median predictor.
- Seasonal naive predictor for suitable time-series data.
- Historical average by a known, available group, if justified.

Do not automatically use a seasonal baseline when the dataset does not support it.

Document why the selected baseline is appropriate.

### Candidate model

Evaluate at least one tree-based model, such as:
- RandomForestRegressor.
- HistGradientBoostingRegressor.
- ExtraTreesRegressor.

Select based on dataset characteristics and practical trade-offs.

Only introduce additional algorithms if they contribute useful comparative evidence.

### Training workflow

Implement a reusable training pipeline that:
1. Loads configuration.
2. Loads and validates data.
3. Creates the appropriate train-validation-test partitions.
4. Fits preprocessing exclusively on training data.
5. Trains the baseline.
6. Trains the selected model.
7. Performs validation or cross-validation when appropriate.
8. Evaluates both models.
9. Records configuration, data assumptions, and results.
10. Saves the selected complete pipeline and metadata.
11. Produces a machine-readable metrics report.

Use a validation set or suitable cross-validation for model selection.

Keep the test set untouched until final model evaluation.

Do not select the best model using test-set performance.

For temporal data, use time-aware validation rather than shuffled cross-validation.

If hyperparameter tuning is used, keep the search space modest, documented, and reproducible. Do not introduce expensive tuning merely for appearances.

## 11. Evaluation and Error Analysis

Use metrics appropriate for order-demand prediction.

At minimum, evaluate:
- MAE.
- RMSE.
- R², where meaningful.

If appropriate, additionally report:
- Median absolute error.
- MAPE only when its limitations around zero or near-zero actual values are addressed.
- WAPE or another clearly defined aggregate percentage error.
- Performance across relevant demand segments.

Explain every metric in plain language and state its units or limitations.

Create a metrics comparison table for the baseline and selected model.

Report performance on held-out observations and distinguish validation metrics from final test metrics.

### Error analysis

Investigate:
- Largest absolute errors.
- Underprediction versus overprediction.
- Errors across low-, medium-, and high-demand observations.
- Performance by time interval, restaurant, or other relevant segment.
- Residual distributions.
- Actual versus predicted values.
- Residuals versus predictions.
- Performance on demand spikes, where the dataset supports that analysis.

Do not claim that a model is superior simply because it has a higher R² if the business-relevant error metrics disagree.

Use plots and tables to explain where the model performs well and where it struggles.

Document practical consequences. For example, systematic underprediction could lead to insufficient staffing or inventory, while overprediction could lead to waste. Present these as possible operational implications, not proven real-world outcomes.

## 12. Model Artifacts and Reproducibility

Save a complete, versioned or clearly identified inference artifact.

Where appropriate, store:
- Fitted preprocessing and model pipeline.
- Feature names and expected input schema.
- Target definition.
- Model configuration.
- Training timestamp in UTC.
- Training-data fingerprint or another suitable provenance identifier.
- Dataset source metadata.
- Package versions.
- Evaluation metrics.
- Model version.
- Validation assumptions.

Use JSON for portable metadata where practical.

Avoid serializing unnecessary training data into the artifact.

Use joblib only with trusted artifacts. Document that loading arbitrary serialized models can execute unsafe code.

Use atomic writes or another sensible approach to avoid leaving corrupted model artifacts after interrupted writes.

Separate the model file from its metadata.

Provide a clear artifact-loading interface that validates compatibility between the requested inputs, metadata, and model.

The repository must not require retraining every time a user requests a prediction.

If trained artifacts are not committed because of size, licensing, or reproducibility considerations, document how to generate them locally.

Do not commit a fake model artifact or a metrics report containing invented results.

## 13. Prediction Interface

Provide a usable inference interface.

Choose either FastAPI or Streamlit based on the project's requirements. Prefer a lightweight implementation that demonstrates real engineering value.

If choosing FastAPI, implement:
- A health endpoint.
- A model-information endpoint.
- A single-prediction endpoint.
- A batch-prediction endpoint, if appropriate.
- Typed request and response schemas.
- Input validation.
- Clear HTTP errors.
- Model loading without retraining on each request.
- Automatic API documentation through the framework.
- Tests for valid and invalid requests.

If choosing Streamlit, implement:
- A clear project title and explanation.
- Input controls derived from the actual model schema.
- Input validation.
- Prediction output.
- Appropriate demand units.
- Model limitations and a short explanation of the result.
- A clean separation between UI code and prediction logic.

Do not hardcode made-up feature values that contradict the dataset schema.

Do not expose training functionality through the public prediction endpoint.

Do not claim that an interface is deployed or publicly available unless it actually is.

### Inference contract

Clearly document the required input fields and expected output.

For a regression model, a response might include:
- Predicted demand.
- Target unit.
- Model version.
- Optional confidence or uncertainty information, but only if a defensible method has been implemented.

Do not label an arbitrary prediction interval as a confidence interval without a statistically appropriate method.

If the target is a count, investigate whether predictions can be negative. If the chosen regression model can produce negative outputs, decide how to handle them and document the consequences. Do not silently clip predictions without justification.

For batch predictions, preserve input-to-output row alignment and validate the entire batch appropriately.

## 14. Command-Line Workflow

Create a professional command-line interface or documented scripts that expose the principal workflows.

Provide straightforward commands for:
- Validating incoming data.
- Generating a data-quality report.
- Running EDA.
- Training the baseline and candidate model.
- Evaluating a trained model.
- Generating predictions from a CSV file.
- Launching the selected interface.
- Running tests and code-quality checks.

Use commands appropriate to the chosen implementation. A Makefile may simplify common workflows, but the Python package should remain usable without Make when possible.

Support configuration and input paths through documented command-line options.

Fail clearly when required files or trained models are missing.

Make training, evaluation, and inference separate operations.

Do not require users to modify source code just to change the dataset path.

## 15. Testing Strategy

Build a meaningful automated test suite using pytest.

Do not rely exclusively on notebook execution or manual checks.

### Unit tests

Test:
- Schema validation.
- Missing required columns.
- Invalid target values.
- Empty input.
- Duplicate observations.
- Numeric and categorical preprocessing.
- Missing-value handling.
- Feature generation.
- Temporal splitting.
- Leakage-sensitive transformations.
- Baseline training.
- Candidate-model training.
- Metrics calculations.
- Model persistence and loading.
- Single predictions.
- Batch predictions.
- Invalid inference schemas.
- Configuration validation.

### Integration tests

Test a complete small workflow using a small synthetic fixture or a clearly designated test dataset:
1. Load test data.
2. Validate the schema.
3. Split the data.
4. Fit preprocessing and a model.
5. Evaluate the model.
6. Save the artifact.
7. Load the artifact.
8. Generate predictions.
9. Validate prediction outputs.

Add API tests if FastAPI is selected.

Tests must not depend on downloading large datasets or accessing external APIs.

Use temporary directories for test artifacts.

Test both successful execution and important failure paths.

Do not write brittle tests that depend on exact floating-point outputs from a particular library version.

Use deterministic fixtures and seeds where appropriate.

## 16. Code Quality and Continuous Integration

Configure Ruff for linting and formatting.

Configure pytest for automated testing.

Add a GitHub Actions workflow that:
- Checks out the repository.
- Installs the supported Python version.
- Installs the project and development dependencies.
- Runs linting.
- Checks formatting.
- Executes the automated test suite.

Use dependency caching where reasonable.

Do not make CI depend on private credentials, a large external dataset, or manual user intervention.

If multiple Python versions are tested, choose versions supported by the declared dependencies.

Include instructions for running all CI-equivalent checks locally.

Resolve the issues discovered during execution instead of merely listing them.

## 17. Configuration, Logging, and Error Handling

Implement sensible configuration management.

Keep meaningful parameters outside business logic, including:
- Dataset paths.
- Target-column configuration.
- Split strategy.
- Random seed.
- Model hyperparameters.
- Artifact output paths.
- Relevant evaluation settings.

Avoid unnecessary configuration for values that will never reasonably change.

Use structured, informative log messages where useful.

Include enough context in error messages for a developer to resolve failures without exposing credentials or sensitive data.

Distinguish between:
- Invalid user input.
- Invalid dataset schema.
- Data-quality problems.
- Missing artifacts.
- Model incompatibility.
- Training failures.
- Inference failures.

Do not catch an exception and silently substitute a dummy prediction.

## 18. Model Monitoring and Operational Limitations

Include a practical discussion of what would be monitored if the system were used operationally.

Potential monitoring dimensions include:
- Input-schema violations.
- Missing-feature rates.
- Feature-distribution changes.
- Prediction-distribution changes.
- Prediction latency.
- Batch failures.
- Delayed ground-truth availability.
- Actual-versus-predicted error once actual demand becomes available.

Implement lightweight drift or prediction diagnostics only if they can be meaningfully demonstrated.

Do not claim to have implemented production monitoring if the repository only contains a written plan.

Clearly distinguish:
1. Implemented functionality.
2. Locally tested functionality.
3. Proposed future production functionality.

Discuss the limitations of historical datasets, changing customer behavior, promotions, special events, missing external variables, and possible differences between offline evaluation and real-world deployment.

## 19. Docker and Deployment

Assess whether Docker adds value for this project.

If included:
- Create a small, reproducible Dockerfile.
- Use an appropriate supported Python base image.
- Avoid running as root where practical.
- Avoid copying secrets or unnecessary datasets into the image.
- Define the appropriate command and port.
- Include a .dockerignore file.
- Document how to build and run the image.
- Make artifact and data locations configurable.

Do not introduce Kubernetes, a cloud infrastructure stack, or distributed serving systems unless they solve a demonstrated requirement.

Provide local execution instructions even if Docker is included.

If deployment is not performed, describe the interface as locally runnable rather than deployed.

## 20. Notebook Quality

Provide a reproducible EDA notebook and, if it adds genuine value, a model-development notebook.

The notebooks must:
- Explain the purpose of each major section.
- Load data through documented paths.
- Use the same underlying project conventions as the Python package.
- Avoid duplicating the entire training pipeline.
- Display useful tables and visualizations.
- Include findings grounded in actual execution.
- Clearly distinguish exploratory experiments from the final reproducible training workflow.

A clean checkout must not depend on hidden notebook state.

The main model should be trainable through the package or command line.

## 21. README.md: Mandatory Comprehensive Documentation

The README is one of the most important deliverables.

It must be detailed enough for a recruiter, another engineer, or an interviewer to understand the project without opening the source code first.

Do not write a generic README with only installation commands and a screenshot.

Include the following sections.

### A. Project title and overview

Explain:
- What the project does.
- Why restaurant demand prediction matters.
- The exact prediction task.
- The primary users or operational stakeholders.
- The technology stack.
- The current implementation status.

### B. Problem statement

Explain the business problem and the consequences of inaccurate demand estimates.

Discuss potential effects on:
- Staffing.
- Ingredient procurement.
- Inventory.
- Food waste.
- Order preparation capacity.

Make it clear that these are potential use cases, not measured benefits unless actual business data supports the claim.

### C. Project objectives and scope

Separate the implemented features from future enhancements.

Explain what the model predicts and what it does not predict.

### D. Dataset documentation

Include:
- Dataset name.
- Original source and access instructions.
- License or usage restrictions.
- Data acquisition process.
- File format.
- Row and column counts, populated only from actual inspection.
- Time coverage, where applicable.
- Target definition.
- Units.
- Important columns.
- Known data-quality limitations.
- Any transformations or exclusions.

Link to the actual dataset source when possible.

### E. Data dictionary

Provide a table with columns such as:

| Field | Data type | Meaning | Role | Availability at prediction time |
|---|---|---|---|---|

Populate this table from the real dataset. Never invent column names or meanings.

### F. Architecture

Provide a high-level architecture diagram using Mermaid, plus a written explanation of every major component.

The diagram should illustrate the actual implementation, including the data ingestion path, validation, preprocessing, training, evaluation, artifact persistence, and inference interface.

Show the primary data flow and the separation between training and prediction.

Also include a repository directory tree and explain the responsibility of each important directory.

The architecture must match the code that actually exists.

### G. End-to-end working

Explain the workflow step by step:

1. Dataset acquisition.
2. Raw data loading.
3. Validation.
4. Exploratory analysis.
5. Feature preparation.
6. Dataset splitting.
7. Baseline training.
8. Candidate-model training.
9. Validation and model selection.
10. Final test evaluation.
11. Artifact saving.
12. Inference-time loading.
13. Input validation.
14. Prediction generation.
15. Results interpretation.

For each stage, explain its inputs, operations, outputs, and important design decisions.

### H. Feature engineering and leakage prevention

Explain:
- How raw columns become model features.
- How categorical and numeric variables are processed.
- Which temporal features are used, if any.
- Which features are excluded and why.
- How future information is prevented from entering training or inference.
- How the same preprocessing is reused at prediction time.

### I. Modeling approach

Explain:
- The selected baseline and why it is appropriate.
- The tree-based model and why it was chosen.
- The preprocessing pipeline.
- The training strategy.
- The validation strategy.
- Hyperparameter selection, if applicable.
- Reproducibility and random seeds.

### J. Evaluation and results

Include a metrics table populated only by actual model execution.

Describe MAE, RMSE, and R², including their interpretation and limitations.

Compare the baseline and final model.

Include links to generated visualizations and the evaluation report.

Explain the main error patterns discovered during actual analysis.

If execution has not produced valid results, mark them as unavailable instead of inventing numbers.

### K. Input and output specifications

Provide the exact inference input schema.

Include a realistic example request using the actual field names and valid example values.

Show an example output and explain each field.

Clearly mark illustrative values as examples and do not present fabricated sample predictions as actual model results.

Document CSV batch input and output behavior, if supported.

Explain validation rules and error responses.

### L. Installation and setup

Provide exact commands for:
- Cloning the repository.
- Creating a virtual environment.
- Activating the environment on Windows and Linux/macOS.
- Installing dependencies.
- Acquiring or configuring the dataset.
- Running validation.
- Running tests.

Verify that the commands match the final repository.

### M. Training and inference commands

Include copy-paste-ready commands for:
- Data validation.
- EDA.
- Training.
- Evaluation.
- Prediction from a file.
- Launching the interface.
- Running quality checks.

Explain the expected artifacts and outputs of each command.

### N. API or application usage

If an API exists, document endpoints, schemas, error responses, and example requests.

If a Streamlit interface exists, document how to launch it and use the available input controls.

Only document functionality that is implemented.

### O. Testing and code quality

Describe the test architecture, important test cases, CI workflow, and how to run all checks locally.

Include actual test status only after running the tests.

### P. Model artifacts and reproducibility

Explain:
- What is saved.
- Where artifacts are stored.
- How metadata is used.
- How to regenerate the model.
- How to reproduce evaluation.
- Which package versions and random seeds matter.
- Why the model and preprocessing pipeline must remain compatible.

### Q. Limitations and assumptions

Discuss the actual limitations of the dataset and model.

Distinguish limitations caused by missing information from limitations of the selected algorithm.

Explain why offline test performance does not automatically guarantee operational success.

### R. Security and responsible data handling

Explain the treatment of credentials, local datasets, serialized model artifacts, and external inputs.

Document relevant security assumptions.

### S. Future improvements

List technically meaningful extensions, such as:
- Time-aware backtesting.
- Promotion and holiday features.
- Probabilistic demand forecasts.
- Prediction intervals with appropriate calibration.
- Drift monitoring.
- Model comparison across additional algorithms.
- Scheduled retraining.
- More robust deployment.

Clearly distinguish future work from implemented functionality.

### T. Resume-ready project summary

Provide two or three concise, defensible resume bullets based on verified implementation.

Do not insert fabricated performance improvements, sample sizes, latency numbers, or accuracy claims.

If actual results support a quantitative claim, explain how it was measured and where the evidence is recorded.

### U. Interview preparation

Include:
- A concise 30-second project explanation.
- A detailed two-minute explanation.
- An explanation of the architecture.
- Why the baseline was selected.
- Why the final model was selected.
- How leakage was prevented.
- Why the chosen metrics matter.
- How the train-validation-test split works.
- What happens when a new prediction request arrives.
- How model persistence works.
- What the largest technical limitations are.
- Potential interviewer questions and technically correct answers.
- Trade-offs and alternatives that were considered.

Answers must reflect the actual code and dataset, not generic ML explanations disconnected from the implementation.

### V. References

List the real dataset source, relevant official documentation, and other sources used.

Do not fabricate references.

## 22. Additional Technical Documentation

Create the following documents when supported by the implementation.

### docs/ARCHITECTURE.md

Explain the component boundaries, dependencies, execution flows, artifact lifecycle, and architectural decisions.

Include an architecture diagram and document how training differs from inference.

### docs/DATA_DICTIONARY.md

Record the actual schema, feature definitions, target, data-quality assumptions, and prediction-time availability.

### docs/MODEL_CARD.md

Document the intended use, training data, evaluation methodology, measured performance, limitations, failure modes, and inappropriate use cases.

### docs/EXPERIMENTS.md

Record the baseline, candidate models, relevant configurations, validation methodology, and measured results.

Use a consistent, reproducible format.

### docs/INTERVIEW_GUIDE.md

Explain the project as an engineer would explain it during a technical interview, including architecture, ML fundamentals, engineering choices, trade-offs, and potential follow-up questions.

Avoid memorized answers that would be contradicted by the implementation.

## 23. Git Hygiene and Repository Readiness

Prepare the repository for publication on GitHub.

Include a suitable .gitignore covering:
- Virtual environments.
- Python bytecode and caches.
- Test and coverage outputs.
- Notebook checkpoints.
- Local environment files.
- Downloaded datasets when appropriate.
- Generated model artifacts when appropriate.
- Temporary files.
- Local logs.

Do not exclude files required to reproduce the project.

Avoid committing large raw datasets unless their license and repository size make that appropriate.

Do not commit unnecessary build products, local absolute paths, secrets, or machine-specific configuration.

Add a license only after selecting an appropriate license and confirming that it is compatible with the project's code and dataset usage. Do not imply that a third-party dataset has been relicensed.

## 24. Required Execution and Verification

Do not finish merely because files have been written.

Execute the project to the extent the environment permits.

At minimum, attempt to:
1. Inspect the final repository structure.
2. Install or resolve dependencies.
3. Validate configuration.
4. Run linting.
5. Check formatting.
6. Run the complete automated test suite.
7. Run the training workflow on the actual dataset when accessible.
8. Generate real evaluation metrics.
9. Save and reload the fitted model.
10. Run prediction on valid sample input.
11. Verify invalid-input handling.
12. Start or test the inference interface.
13. Check that README commands correspond to real entry points.
14. Verify that generated documentation matches the actual implementation.

Fix failures caused by your implementation.

If a check cannot be completed because of missing network access, missing credentials, unavailable data, or environment limitations, identify the exact limitation and provide the command needed to run that check later.

Never claim that a command passed if it was not executed successfully.

Never claim that the system is production-deployed merely because it can run locally.

## 25. Definition of Done

The project is complete only when the following deliverables exist and have been verified to the extent possible.

- A professional, modular Python repository.
- A documented, legitimate dataset source.
- A working data-ingestion and validation workflow.
- Reproducible EDA and meaningful visualizations.
- A defensible feature-engineering process.
- A leakage-aware evaluation strategy.
- A simple baseline model.
- At least one tree-based model.
- Baseline-versus-model evaluation.
- Error analysis.
- A persisted model pipeline and metadata.
- A functioning prediction interface or API.
- A documented command-line workflow.
- Automated unit and integration tests.
- Linting and formatting configuration.
- GitHub Actions CI.
- A comprehensive README covering architecture, workflow, inputs, outputs, installation, results, limitations, and interview preparation.
- Additional architecture, data dictionary, model card, experiment, and interview documentation.
- Accurate execution and verification results.
- Resume bullets grounded in implemented functionality.

Do not sacrifice correctness or maintainability to make the project appear more advanced than it is.

A smaller system that genuinely works, is reproducible, and is thoroughly explained is better than a complex system filled with placeholders.

## 26. Final Response Requirements

After implementation, provide a concise but informative completion report containing:

1. Project overview.
2. Final repository structure.
3. Dataset chosen and source.
4. Architecture and key engineering decisions.
5. Baseline and candidate models implemented.
6. Actual measured evaluation results, if available.
7. Automated test results.
8. Linting and formatting results.
9. Commands to install, train, evaluate, and run predictions.
10. Commands to launch the interface.
11. Links to the README and major documentation files.
12. Known limitations and incomplete items.
13. Any checks that could not be performed, with reasons.
14. Two or three defensible resume bullets based on the completed project.

Clearly separate completed, tested, and unverified functionality.

Do not claim that the project is complete if required components are missing. Explain what remains and why.

Begin by inspecting the working environment and planning the implementation. Then build the actual project, execute it, verify it, and document it comprehensively.