# UWebASR Confidence Calibration

## Motivation and Use Case

The primary goal is to provide a reproducible set of open scripts that can build
an accuracy regressor for any user-provided evaluation set and any selected
UWebASR model. A user should be able to run the calibration locally against
their own labelled data and obtain a model-specific predictor of recognition
accuracy, without sharing their reference transcripts with the UWebASR team.

Only audio is sent to the selected UWebASR endpoint for recognition. References
remain on the user's machine and are used locally to compute training targets,
train/test splits, calibration features, validation metrics, and the final
regressor. This makes the workflow suitable for private evaluation sets while
still allowing users to calibrate confidence-derived accuracy estimates for the
exact ASR endpoint and model they intend to use.

This document defines the current calibration procedure for estimating word
recognition accuracy from UWebASR/SpeechCloud CTC confidence streams. It is
intended as the implementation basis for a script which takes one UWebASR
endpoint and one labelled evaluation set, runs recognition, builds a calibration
dataset, trains a model, and returns an accuracy predictor.

The same document also specifies the feature extractor that should operate on
the final `ctc_*` objects contained in an `asr_result`.

## Command-Line Contract

The calibration script should expose a single end-to-end command with all
parameters saved to the output directory:

```text
uwebasr-calibrate \
  --dataset DATASET_MANIFEST \
  --uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer \
  --output-dir OUTPUT_DIR \
  --target-segments 8000 \
  --jobs 6 \
  --seed 13
```

The `--uwebasr-url` argument is the complete model endpoint URL. Users normally
pass it without the `format` query parameter. The script requests:

```text
format=speechcloud_json
```

If the supplied URL has no query string, append `?format=speechcloud_json`. If
it already has other query parameters, append `&format=speechcloud_json`. If it
already contains a `format` parameter, its value must be `speechcloud_json`;
otherwise the script should fail with a clear error.

The command should be resumable. The implementation may keep intermediate
recognition results internally, but that storage format is not part of the
public user-facing contract.

## Dataset Manifest

The primary input is a UTF-8 dataset manifest in JSONL, CSV, or JSON-array
format. After parsing, every row must provide:

```text
utt_id      stable utterance identifier
audio_path  path to the audio file
reference   reference transcript
```

Optional columns:

```text
speaker_id  explicit speaker/group identifier
video_id    explicit long-recording/group identifier for test_real
```

If `utt_id` is missing, it may be derived from the audio filename stem, but the
script must fail if the resulting IDs are not unique. Relative `audio_path`
values are resolved relative to the manifest directory. Missing audio files,
empty references, duplicate utterance IDs, and invalid encodings are fatal input
errors unless the user explicitly enables a skip-bad-rows option.

Example JSONL row:

```json
{
  "utt_id": "12345_A_001_000001",
  "audio_path": "audio/12345_A_001_000001.wav",
  "reference": "the reference transcript",
  "speaker_id": "12345",
  "video_id": "12345_A_001"
}
```

For compatibility with the MALACH-style local datasets, a JSON array with
`filename` and `text` fields is also accepted and mapped as:

```text
utt_id     = stem(filename)
audio_path = dataset_dir / "audio" / basename(filename)
reference  = text
```

## Objective

For each evaluated speech segment, let

```text
Acc = max(0, 1 - edit_errors / reference_words).
```

The edit distance is computed after the text normalization defined below.
References are required only during training and evaluation. At inference time,
the predictor must estimate accuracy from ASR confidence features only:

```text
Acc_hat = f(CTC confidence features).
```

The preferred interpretation of `Acc_hat` is direct expected recognition
accuracy. Therefore the trained predictor should be evaluated both by prediction
error and by the shape of the scatter plot `Acc_hat` versus `Acc`.

## Text Normalization

The reference and hypothesis must be normalized identically before word
counting and edit-distance computation. The default calibration normalizer is:

```text
text = lowercase(text)
tokens = all Unicode word tokens matching:
         [^\W_]+(?:['’][^\W_]+)?
```

This keeps Unicode letters and digits, removes underscores and standalone
punctuation, and preserves internal straight or curly apostrophes. It does not
perform stemming, diacritic stripping, number expansion, or language-specific
rewriting. Empty token sequences are invalid for training targets.

Edit distance is standard Levenshtein distance over the normalized token
sequence. The reimplementation should use `jiwer` for word-error computation
and, where needed, word-level alignment. The normalizer name, `jiwer` version,
and transformation pipeline must be stored in `config.json` and model metadata.

## Recognition

The recognition stage takes a complete UWebASR endpoint URL and a labelled
dataset containing audio paths, utterance IDs, and references.

The recognizer must request `speechcloud_json` output. The implementation may
store recognition results locally so that re-running feature extraction or model
training does not require a new ASR pass.

Only final ASR results are used:

```text
type == "asr_result"
partial_result == false
```

The final result must contain complete CTC data:

```text
ctc_tokens
ctc_probs
ctc_frame_len
```

The `ctc_probs` stream is used including `<blk>` tokens. Missing or incomplete
CTC data is a recognition error; such entries must be recognized again or the
workflow must stop. They must not be imputed or silently interpreted as low
confidence.

The stored recognition result must retain final word-level data:

```text
word_array
word_times
```

These fields are required to cut long utterances into word-aligned CTC spans.
For the current calibration procedure, a result missing `word_array` or
`word_times` is invalid and the workflow must stop or recognize the utterance
again. No CTC-only fallback is part of this specification.

## UWebASR Endpoint Call

The script calls one complete UWebASR model endpoint directly through the HTTP
API. The canonical endpoint argument is:

```text
--uwebasr-url https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer
```

The request URL is formed by adding `format=speechcloud_json` when it is not
already present:

```text
https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer?format=speechcloud_json
```

For a development SpeechCloud endpoint, the complete URL can point to
`speechcloud-dev`:

```text
https://uwebasr.zcu.cz/api/v2/speechcloud-dev/generic/en/zipformer
```

The request is an HTTP `POST`. The body is the audio stream itself, not a JSON
request object. The current implementation converts every input audio file to
16 kHz mono Ogg/Vorbis before sending it:

```bash
ffmpeg -xerror -hide_banner -loglevel error \
  -i INPUT_AUDIO \
  -ar 16000 -ac 1 -vn \
  -c:a libvorbis -q:a 10 \
  -f ogg -
```

A minimal equivalent API call is:

```bash
curl -sS \
  -X POST \
  -H 'Content-Type: audio/ogg' \
  --data-binary @audio.ogg \
  'https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer?format=speechcloud_json'
```

The response body is a SpeechCloud JSON document, typically a JSON array of
events/results. Implementations may store this raw response together with the
utterance metadata for reproducibility, for example:

```json
{
  "utt_id": "utterance-id",
  "audio_path": "/path/to/audio",
  "reference": "reference transcript",
  "endpoint_url": "https://uwebasr.zcu.cz/api/v2/speechcloud/generic/en/zipformer",
  "result_format": "speechcloud_json",
  "result": [
    "... raw SpeechCloud JSON response ..."
  ]
}
```

The UWebASR response format requested from the endpoint is always
`speechcloud_json`. Any local storage format for recognized utterances is an
implementation detail.

The implementation should reuse HTTP session cookies across requests handled by
the same worker. This allows the server to reuse worker-side state. If
recognition is parallelized, each client worker should keep its own cookie jar.

Default operational parameters:

```text
jobs = 6
timeout_seconds = 90
retries = 7
backoff = min(30, 2 ** (attempt - 1)) seconds
```

HTTP 503 and transient network/parse failures are retried. Permanent HTTP
errors, repeated timeouts, ffmpeg conversion failures, invalid JSON responses,
and missing final CTC after all retries are reported. The main workflow should
stop before training if any required utterance is still missing a valid
recognition result.

The script must validate every recognition result before feature extraction. A
valid result contains exactly one final ASR result with complete CTC arrays:

```text
type == "asr_result"
partial_result == false
ctc_tokens: list
ctc_probs: list
len(ctc_tokens) == len(ctc_probs)
```

If `ctc_frame_len` and `audio_duration` are present, the implementation should
also check that the number of CTC frames is consistent with
`round(audio_duration / ctc_frame_len)`, allowing a tolerance of two frames.
`ctc_frame_len` and `audio_duration` are measured in seconds.

## Evaluation Rows

After recognition, each utterance is converted into an evaluation row containing
at least:

```text
utt_id
reference
hypothesis
reference_words
edit_errors
final CTC stream
```

The `hypothesis` is taken from the final ASR result. `reference_words` and
`edit_errors` are computed after the same word normalization. Rows with no
reference words are discarded.

The target accuracy calculation should be implemented through `jiwer` so that
the same transformation pipeline is used for word counts, edit errors, and any
alignment objects needed by later segmentation. For every evaluation row:

```text
reference_words = number of transformed reference words
edit_errors     = substitutions + insertions + deletions from jiwer alignment
accuracy        = max(0, 1 - edit_errors / reference_words)
```

The final hypothesis text is reconstructed from the final `word_array`. The
expected final CTC-related fields are:

```text
ctc_tokens: list[str]
ctc_probs: list[float] in probability scale, not log-probabilities
ctc_frame_len: float seconds per CTC frame
word_array: list[str]
word_times: list[[start_seconds, end_seconds]]
```

If a manifest does not supply `speaker_id`, the speaker ID is extracted from
`utt_id` using this regular expression:

```text
(?<!\d)(\d{4,5})(?!\d)
```

The first match is used and zero-padded to five digits. This is the group key
for train/test separation.

Examples for MALACH-style utterance IDs:

```text
12345_A_001_000123  -> speaker_id = 12345
12345_01_000123     -> speaker_id = 12345
malach_1234_seg001  -> speaker_id = 01234
```

If a manifest supplies `speaker_id`, it takes precedence. If neither
`speaker_id` nor a numeric speaker ID can be found, the script should fail by
default because speaker-disjoint evaluation cannot be guaranteed. A user may
explicitly choose `--split-group utterance` for datasets where speaker
separation is impossible, but reports must mark this as a weaker split.

The video ID for `test_real` is taken from manifest `video_id` when present.
Otherwise use the first matching utterance-ID pattern:

```text
^(\d{5}_[A-Z]_\d{3})_
^(\d{5}_\d{2})_
```

If no pattern matches, fall back to the source utterance ID without any segment
suffix.

## Train, Test, and Test-Real Split

The train/test split is performed before segmentation and before ensemble
sampling. The split is group-disjoint by speaker, using:

```text
train_fraction = 0.8
seed = 13
```

The split must be deterministic for a fixed seed and must be performed by group,
not by utterance, when speaker IDs are available. The script must save the exact
train and test group lists. No speaker may appear in both train and test. This
constraint also applies to derived segments and ensemble samples, because they
are generated only after the speaker split. If fewer than two groups are
available, the workflow must stop unless the user explicitly requests an
utterance-level split.

The held-out speaker partition is used in two ways:

```text
test
```

Synthetic ensemble samples generated by the same procedure as training samples.

```text
test_real
```

Realistic evaluation points built from held-out speaker/video material. Each
speaker/video group is processed with 512-word windows and then aggregated back
to one score per speaker/video by a reference-word-weighted mean of the window
predictions.

Hyperparameter search is allowed to use only the training partition and an
internal validation split. The `test` and `test_real` partitions are strictly
held out until final evaluation.

## Word-Aligned Segmentation

Utterances are first converted into shorter word-aligned segments. This gives
more training material and reduces the mismatch caused by highly variable
utterance lengths.

For each utterance, generate several random segmentations. The number of
segmentation variants is not a fixed language constant in the final script. It
is derived from an approximate target segment count supplied by the user:

```text
target_segments_per_language ~= 8000
```

This is the default value for `--target-segments`.

Here, "balanced" means that the script adjusts the number of random
segmentation variants so that each language or calibration set has
approximately this number of word-aligned segments after segmentation, not the
same number of original utterances. The current working target corresponds to:

```text
about 8000 total segments per language
about 6000 training segments per language
```

The implementation should estimate the number of variants from a short
pre-segmentation pass. A practical rule is:

```text
estimated_segments_per_variant =
    number of accepted word-aligned segments produced by one random segmentation

segment_variants =
    max(1, round(target_segments_per_language / estimated_segments_per_variant))
```

The script should then perform the actual segmentation with this value and
report the resulting train, test, and total segment counts. With the current
evaluation data and speaker split, the historical variant counts and resulting
segment counts were:

```text
lang  variants  train_segments  test_segments  total_segments
cs    4         6744            1728           8472
de    2         6088            2065           8153
en    11        6365            1749           8114
sk    21        6174            1764           7938
```

For a new dataset or language, these historical variant counts should be
treated only as examples of the target-driven rule.

The target segment lengths are sampled between:

```text
min_segment_words = 10
max_segment_words = 256
```

The segmentation is performed over complete word spans. For a reference word
span, the corresponding hypothesis span is estimated by word-level alignment
between normalized reference and hypothesis tokens. The hypothesis span is then
mapped to word times, and the CTC frame interval is obtained from the time span
and `ctc_frame_len`.

The implementation must define a deterministic word-alignment procedure from
normalized reference words to normalized hypothesis words. For a reference span
`[ref_start, ref_end)`, it must derive the corresponding hypothesis span using
the alignment. The policy for substitutions, insertions, and deletions must be
documented in the run metadata. The segment time span is then obtained by
mapping the selected hypothesis words to final ASR word times.

A valid implementation must guarantee:

```text
the segment contains complete reference words
the segment target accuracy is computed from the aligned hypothesis span
the CTC slice covers the time span of the aligned hypothesis words
the alignment policy is deterministic for a fixed input
```

CTC frames should be selected consistently from the resulting time span, for
example by frame centers:

```text
center_i = (i + 0.5) * ctc_frame_len
keep frame i iff start_time <= center_i < end_time
```

Segments are discarded if they do not contain all three required CTC streams:

```text
ctc_all
ctc_blank
ctc_nonblank
```

These streams are derived after time cutting from `ctc_tokens` and `ctc_probs`;
they are not expected to be present as native SpeechCloud fields.

Segments are also discarded if the CTC span is implausibly long:

```text
max_ctc_frames_per_word = 80
```

This guard catches alignment failures and incomplete timing information.

Random segmentation should use the global run seed and must be deterministic for
the same input manifest and configuration. Each segment must contain between
`min_segment_words` and `max_segment_words`, except that a short final remainder
may be merged into the previous segment or dropped according to a policy saved
in the run metadata.

For `test_real` windowing, utterances longer than the target window size are cut
into deterministic word-aligned chunks. Shorter utterances are kept and joined
with neighbouring held-out utterances from the same speaker/video group.

## Ensemble Sampling

The model is trained on ensemble samples rather than on individual utterances.
An ensemble sample is a set of word-aligned segments sampled with replacement
until it contains enough reference words:

```text
min_words = 512
min_segments = 2
```

The target accuracy for an ensemble is computed exactly from accumulated edit
errors and accumulated reference words:

```text
Acc = max(0, 1 - sum(edit_errors) / sum(reference_words)).
```

The current sampling strategy is accuracy-decile mixture sampling. Within each
train or test partition, eligible segments are sorted by their true segment
accuracy and divided into ten deciles. For sample `i`, the primary decile is:

```text
primary_decile = i mod 10
```

Each segment draw is then taken from:

```text
75% probability: primary_decile
25% probability: all other deciles
```

This creates a broad but smooth accuracy distribution. It avoids isolated
quartile clusters while still increasing the density of low- and high-error
examples.

The current sample counts are:

```text
train = 64000
test  = 16000
```

These numbers are per endpoint/dataset calibration run. The future end-to-end
script may expose them as parameters, but these values are the current default.
Samples are generated with replacement, so the configured counts are produced
even when the held-out segment pool is small. If fewer than ten non-empty
accuracy deciles can be formed, use the available non-empty quantile pools and
record the reduced number in the summary.

The train and test ensemble samplers must use independent deterministic random
streams derived from the run seed.

## CTC Feature Streams

For every ensemble sample, CTC probabilities are concatenated across all chosen
segments. The feature extractor then builds three probability streams:

```text
ctc_all       = all CTC probabilities, including <blk>
ctc_blank     = probabilities whose token is <blk>
ctc_nonblank  = probabilities whose token is not <blk>
```

Absolute counts are retained as metadata and diagnostics only:

```text
ctc_all_count
ctc_blank_count
ctc_nonblank_count
```

They must not be used as model inputs. Length-dependent information should be
represented only by ratios or normalized statistics, so that the predictor can
be applied to utterances or windows of different lengths.

The count-derived model features are:

```text
ctc_blank_fraction          = blank_count / all_count
ctc_nonblank_fraction       = nonblank_count / all_count
ctc_blank_to_nonblank_ratio = blank_count / max(1, nonblank_count)
ctc_nonblank_to_blank_ratio = nonblank_count / max(1, blank_count)
ctc_blank_log_ratio         = log((blank_count + 1) / (nonblank_count + 1))
```

## Feature Extractor

The full compact feature pool contains distributional statistics for
`ctc_all`, `ctc_blank`, and `ctc_nonblank`, plus normalized CTC blank/nonblank
structure features. It remains useful for feature search and diagnostics.

The current deployable predictor uses the top 20 features selected by
validation permutation importance from the 75/25 decile-mixture experiment:

```text
ctc_blank_mean_run_fraction
ctc_nonblank_error_geom_mean
ctc_nonblank_to_blank_ratio
ctc_blank_p040
ctc_nonblank_mean_run_fraction
ctc_nonblank_harmonic_mean
ctc_blank_range
ctc_nonblank_p001
ctc_blank_run_len_cv
ctc_blank_log_ratio
ctc_nonblank_error_mean
ctc_blank_p030
ctc_nonblank_p070
ctc_nonblank_geom_mean
ctc_nonblank_frac_lt_50
ctc_nonblank_short_run_fraction
ctc_blank_p000
ctc_blank_short_run_fraction
ctc_blank_neglog_error_p50
ctc_blank_max_run_fraction
```

The feature extractor should compute these features directly from the final
`ctc_tokens` and `ctc_probs` arrays in `asr_result`. The required input is an
ordered sequence of token/probability pairs:

```text
[(ctc_tokens[0], ctc_probs[0]), ..., (ctc_tokens[n-1], ctc_probs[n-1])]
```

Only final `asr_result` objects are valid inputs. All probabilities must be
finite and clipped before feature extraction:

```text
eps = 1e-9
p = min(1 - eps, max(eps, raw_probability))
```

The extractor constructs the following streams:

```text
all_values      = [p_i for all tokens]
blank_values    = [p_i where token_i == "<blk>"]
nonblank_values = [p_i where token_i != "<blk>"]
blank_mask      = [token_i == "<blk>"]
```

If any of `all_values`, `blank_values`, or `nonblank_values` is empty, the
segment/window is invalid for this model. If a feature calculation produces NaN
or infinity, the row is invalid before model training. Prediction-time handling
of non-finite standardized values must match the training pipeline and be
recorded in the model metadata.

### Distribution Features

The following definitions are applied to either `blank_values` or
`nonblank_values`, depending on the feature prefix. Let `values` be the selected
stream and `sorted_values = sort(values)` in ascending order.

Nearest-rank probability percentiles are used for the `pXXX` features:

```text
nearest_rank_percentile(sorted_values, q):
    threshold = (q / 100) * len(sorted_values)
    index = ceil(threshold) - 1
    index = clamp(index, 0, len(sorted_values) - 1)
    return sorted_values[index]
```

Thus:

```text
p000 = nearest_rank_percentile(values, 0)
p001 = nearest_rank_percentile(values, 1)
p030 = nearest_rank_percentile(values, 30)
p040 = nearest_rank_percentile(values, 40)
p070 = nearest_rank_percentile(values, 70)
p100 = nearest_rank_percentile(values, 100)
range = p100 - p000
```

The top-20 model uses the following distribution features:

```text
ctc_blank_p000   = p000(blank_values)
ctc_blank_p030   = p030(blank_values)
ctc_blank_p040   = p040(blank_values)
ctc_blank_range  = p100(blank_values) - p000(blank_values)

ctc_nonblank_p001       = p001(nonblank_values)
ctc_nonblank_p070       = p070(nonblank_values)
ctc_nonblank_geom_mean  = exp(mean(log(nonblank_values)))
ctc_nonblank_error_mean = mean(1 - nonblank_values)
```

The harmonic mean is:

```text
ctc_nonblank_harmonic_mean =
    len(nonblank_values) / sum(1 / nonblank_values)
```

The geometric mean of the nonblank error stream is:

```text
nonblank_errors = 1 - nonblank_values

ctc_nonblank_error_geom_mean =
    exp(mean(log(max(eps, nonblank_errors))))
```

The threshold fraction is:

```text
ctc_nonblank_frac_lt_50 =
    mean(nonblank_values < 0.50)
```

The blank negative-log-error median uses a linear quantile of
`-log(1 - p)`, matching the broad feature implementation:

```text
blank_errors = max(eps, 1 - blank_values)
blank_neglog_errors = -log(blank_errors)

ctc_blank_neglog_error_p50 =
    linear_quantile(blank_neglog_errors, 0.50)
```

The implementation should use the same quantile convention as NumPy
`quantile(..., method="linear")` for this feature.

### Count-Ratio Features

The top-20 model uses two count-derived ratio features:

```text
all_count = len(all_values)
blank_count = len(blank_values)
nonblank_count = len(nonblank_values)

ctc_nonblank_to_blank_ratio =
    nonblank_count / max(1, blank_count)

ctc_blank_log_ratio =
    log((blank_count + 1) / (nonblank_count + 1))
```

Absolute counts are not model inputs.

### CTC Run-Structure Features

CTC structure features are computed from `blank_mask`, not from probability
values. A run is a maximal contiguous sequence of identical blank/nonblank
states. Let:

```text
n_frames = len(blank_mask)
blank_run_lengths = lengths of runs where blank_mask is true
nonblank_run_lengths = lengths of runs where blank_mask is false
```

For an empty run list, the corresponding feature is `0.0`.

The top-20 model uses:

```text
ctc_blank_mean_run_fraction =
    mean(blank_run_lengths) / n_frames

ctc_nonblank_mean_run_fraction =
    mean(nonblank_run_lengths) / n_frames

ctc_blank_max_run_fraction =
    max(blank_run_lengths) / n_frames

ctc_blank_run_len_cv =
    std(blank_run_lengths) / mean(blank_run_lengths)

ctc_blank_short_run_fraction =
    mean(blank_run_lengths <= 2)

ctc_nonblank_short_run_fraction =
    mean(nonblank_run_lengths <= 2)
```

The coefficient of variation must use the same standard-deviation convention in
training and inference. If the mean run length is zero, the coefficient of
variation is `0.0`.

### Top-20 Feature Vector Order

The serialized model should store the feature list explicitly. The current
feature vector order is:

```text
[
  ctc_blank_mean_run_fraction,
  ctc_nonblank_error_geom_mean,
  ctc_nonblank_to_blank_ratio,
  ctc_blank_p040,
  ctc_nonblank_mean_run_fraction,
  ctc_nonblank_harmonic_mean,
  ctc_blank_range,
  ctc_nonblank_p001,
  ctc_blank_run_len_cv,
  ctc_blank_log_ratio,
  ctc_nonblank_error_mean,
  ctc_blank_p030,
  ctc_nonblank_p070,
  ctc_nonblank_geom_mean,
  ctc_nonblank_frac_lt_50,
  ctc_nonblank_short_run_fraction,
  ctc_blank_p000,
  ctc_blank_short_run_fraction,
  ctc_blank_neglog_error_p50,
  ctc_blank_max_run_fraction
]
```

The training, validation, test, and inference code must use exactly this stored
order when constructing the model input matrix.

## Model Training

The canonical estimator is:

```text
sklearn.ensemble.HistGradientBoostingRegressor(loss="absolute_error")
```

The model is trained separately for each endpoint/dataset calibration set. The
target is ensemble accuracy.

The training lifecycle is:

1. Split the `train` ensemble samples into a train-internal fit subset and
   validation subset.
2. For each hyperparameter setting, fit preprocessing and HGBR on the fit
   subset only, then evaluate MAE on the validation subset.
3. Select the hyperparameters with the lowest validation MAE.
4. Refit preprocessing and HGBR on the complete `train` ensemble partition.
5. Fit the affine calibration on predictions from this final model on the
   complete `train` ensemble partition.
6. Evaluate the calibrated predictor on `validation`, `test`, and `test_real`
   without refitting anything on those targets.

Feature standardization statistics are always computed from the data used to
fit the corresponding model. The final serialized predictor stores the
standardization statistics fitted on the complete `train` ensemble partition.

The affine calibration has the form:

```text
Acc_hat = clip(a + b * HGBR(features), 0, 1)
```

The affine calibration is part of the trained predictor. It is not fit on test
or test-real targets.

The implementation artifact should store at least:

```text
endpoint identifier
feature list and feature version
feature standardization mean and scale
HGBR model parameters and fitted trees
affine calibration intercept a
affine calibration slope b
training metadata and random seed
```

The recommended artifact layout is a directory:

```text
model/
  model.joblib
  metadata.json
```

`model.joblib` contains the trained regressor and any preprocessing objects
needed for prediction. `metadata.json` contains the endpoint identifier,
feature list and order, feature version, normalizer version, training
configuration, selected hyperparameters, affine calibration parameters,
software versions, and validation score used for model selection.

## Hyperparameter Search

Hyperparameters are selected using an internal validation split from the
training partition. The held-out `test` and `test_real` sets are not used for
model selection.

Current search grid:

```text
learning_rate       in {0.02, 0.05}
max_leaf_nodes      in {15, 31}
min_samples_leaf    in {40, 200}
l2_regularization   in {0.0, 0.1}
max_iter            = 700
loss                = absolute_error
validation_fraction = 0.2  # train-internal split for hyperparameter selection
seed                = 13
```

Any estimator-internal early stopping must be reported in `metadata.json`,
including its validation fraction if enabled. This internal early-stopping split
is distinct from the train-internal validation subset used for hyperparameter
selection.

The best setting is chosen by lowest validation MAE, with RMSE and correlation
used only as secondary diagnostics.

Primary metric:

```text
MAE = mean(abs(Acc - Acc_hat))
```

Secondary diagnostics:

```text
RMSE
Pearson correlation
identity-line calibration residuals
scatter plot shape
test_real video-level MAE
test_real video-level Pearson correlation
```

The training script must report at least MAE and Pearson correlation for all
three evaluation partitions:

```text
validation
test
test_real
```

For `validation`, metrics are computed on the train-internal validation split
used for hyperparameter selection. For `test`, metrics are computed on the
held-out synthetic ensemble samples. For `test_real`, metrics are computed
after aggregating 512-word window predictions to speaker/video-level points.

If Pearson correlation is undefined because either side has zero variance or
too few points, report it as `null` in JSON and `nan` in CSV.

## Prediction and Windowing

For ordinary held-out ensemble samples, prediction is direct:

```text
features -> standardization -> HGBR -> affine calibration -> Acc_hat
```

For realistic long recordings, prediction is performed by windowing. A held-out
speaker/video group is converted into approximately 512-reference-word windows.
Utterances longer than 512 words are cut into word-aligned 512-word chunks.
Shorter utterances are accumulated in order until the window reaches the target
length. A final remainder is retained if it contains at least 10 reference
words.

Each window is predicted independently. The final speaker/video-level estimate
is the reference-word-weighted mean of its window predictions:

```text
Acc_hat_video =
    sum(window_ref_words * Acc_hat_window) / sum(window_ref_words)
```

The corresponding ground-truth video accuracy is computed by the same weighted
aggregation or, equivalently, by summing edit errors and reference words over
the same windows.

At inference time, when references are not available, the same windowing
principle should be applied using available ASR word timing or a fixed CTC/time
window. The model input remains the CTC-derived feature vector for each window,
and the output is averaged using the best available duration or word-count
proxy. For supervised evaluation, reference-word weights are used.

## Artifacts and Reports

The output directory should contain stable, documented files so that downstream
users can inspect and reproduce the calibration:

```text
config.json
utterance_metrics.csv
segments.csv
features.csv
predictions.validation.csv
predictions.test.csv
predictions.test_real_window.csv
predictions.test_real.csv
metrics.json
model/
plots/
```

Required columns for `utterance_metrics.csv`:

```text
utt_id
speaker_id
video_id
audio_path
reference_words
hypothesis_words
edit_errors
accuracy
```

Required columns for prediction CSV files:

```text
sample_id
split
accuracy
estimated_accuracy
residual
ref_words
```

For `predictions.test_real.csv`, `sample_id` is the speaker/video group ID and
the file should also include `n_windows`. `metrics.json` must include MAE and
Pearson correlation for `validation`, `test`, and aggregated `test_real`, plus
the number of points used in each metric.

`config.json` must record all user-provided arguments, random seeds, endpoint
URL, target segment count, selected split groups, normalizer version, feature
version, package versions, and the generated train/test speaker or group lists.

## End-to-End Script Contract

The future calibration script should implement the following contract:

```text
input:
  UWebASR endpoint URL
  labelled evaluation dataset
  approximate target number of word-aligned segments
  output directory

output:
  utterance-level metrics table
  train/test/test_real feature tables
  trained accuracy predictor
  validation/test/test_real reports with MAE and Pearson correlation
  prediction CSV files
  scatter plots
```

The script should execute these stages:

1. Recognize all utterances through the endpoint.
2. Validate that each final result contains complete CTC data.
3. Compute normalized reference/hypothesis metrics.
4. Split rows by speaker into train and held-out test.
5. Estimate the number of segmentation variants from the target segment count.
6. Generate word-aligned segment variants and report the resulting counts.
7. Generate 512-word ensemble samples with 75/25 accuracy-decile mixture sampling.
8. Extract the top-20 CTC confidence features.
9. Train HGBR models with train-internal hyperparameter search.
10. Fit the affine calibration on training predictions.
11. Report MAE and Pearson correlation on `validation`, `test`, and aggregated
    `test_real`.
12. Save the trained predictor and all reports.

The workflow should be resumable. If recognition has already succeeded and CTC
data are complete, downstream feature extraction and training should be
repeatable without contacting the ASR endpoint again.

## Current Reference Result

Using the top-20 feature set, 75/25 decile-mixture sampling, 64k training
samples, 16k test samples, speaker-disjoint split, and 512-word test-real
windowing, the current held-out segment-level results are:

```text
lang  validation_MAE  test_MAE  test_corr
cs    0.01917         0.03150   0.8246
de    0.01863         0.03380   0.8798
en    0.01605         0.02867   0.9000
sk    0.01307         0.02602   0.7518
```

The corresponding aggregated `test_real` results are:

```text
lang  test_real_MAE  test_real_corr
cs    0.03589        0.8126
de    0.04155        0.9629
en    0.02468        0.8427
sk    0.05592       -0.0557
```

These numbers are not a formal benchmark; they are the current implementation
target for regression testing future scripts.
