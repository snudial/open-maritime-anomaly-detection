import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from omad.score.infer import infer_anomaly_type_from_filename, infer_expected_T_from_path
from omad.score.llm_core import generate_json_validated, load_model, set_hf_cache, validate_score_payload
from omad.score.logging_utils import get_arg_value, iter_query_files, log, sanitize_user_query_text, save_output_json

CACHE_DIR = "./data/models"
MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_LOG_DIR = str(Path(__file__).resolve().parent / "logs")
DEFAULT_MAX_NEW_TOKENS = 1024
# -1 = retry until the model produces a valid payload. Every query must end up
# with a score, so the batch never gives up on one.
DEFAULT_MAX_RETRIES = -1
# Escape hatch for the one failure mode retrying cannot fix: a model that keeps
# emitting T+/-N scores. After this many attempts the length is repaired in code.
DEFAULT_COERCE_AFTER = 20

# Lazy model loading - only load when actually needed
_tokenizer = None
_model = None


def _get_model():
    """Load model on first use (lazy initialization)."""
    global _tokenizer, _model
    if _tokenizer is None:
        set_hf_cache(CACHE_DIR)
        _tokenizer, _model = load_model(MODEL_ID, CACHE_DIR)
    return _tokenizer, _model


def _find_valid_existing_output(
    *,
    out_dir: str,
    source_name: str,
    anomaly_type: str,
    expected_T: Optional[int],
) -> Optional[Path]:
    """Return an already-saved output for this query that passes validation.

    Outputs from failed runs (parse_error wrappers, wrong scores length) are
    ignored so that a resumed run re-scores exactly the queries that need it.
    """
    out_path = Path(out_dir)
    if not out_path.is_dir():
        return None
    for fp in sorted(out_path.glob(f"qwen_output_{source_name}_*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        ok, _ = validate_score_payload(
            payload,
            expected_anomaly_type=anomaly_type,
            expected_T=expected_T,
        )
        if ok:
            return fp
    return None


def _parse_anomaly_type_flag(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().upper()
    return v if v in {"A1", "A2"} else None


def read_multiline(prompt: str, log_fh=None) -> str:
    log(prompt, log_fh=log_fh)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() in {":q", ":quit", ":exit"}:
            return ":quit"
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _batch_from_root(
    *,
    batch_root: str,
    out_dir: str,
    max_new_tokens: int,
    max_retries: int,
    log_fh,
    resume: bool = True,
    coerce_after: int = DEFAULT_COERCE_AFTER,
) -> int:
    tokenizer, model = _get_model()
    root = Path(batch_root)
    files = sorted(root.glob("**/user_query/*.txt"))
    log(f"[BATCH] batch_root={batch_root} (files={len(files)})", log_fh=log_fh)
    log(f"[BATCH] out_dir={out_dir} resume={resume}", log_fh=log_fh)

    resumed_cnt = 0
    failed: list[str] = []
    for i, fp in enumerate(files, start=1):
        raw = fp.read_text(encoding="utf-8", errors="replace")
        user_query = sanitize_user_query_text(raw)
        if not user_query:
            log(f"[BATCH {i}/{len(files)}] SKIP empty: {fp}", log_fh=log_fh)
            continue

        anomaly_type = infer_anomaly_type_from_filename(fp)
        expected_T = infer_expected_T_from_path(fp)
        if anomaly_type is None:
            log(
                f"[BATCH {i}/{len(files)}] SKIP (cannot infer anomaly type from filename): {fp.name}",
                log_fh=log_fh,
            )
            continue
        if anomaly_type == "A3":
            log(f"[BATCH {i}/{len(files)}] SKIP A3: {fp.name}", log_fh=log_fh)
            continue

        if resume:
            done = _find_valid_existing_output(
                out_dir=out_dir,
                source_name=fp.stem,
                anomaly_type=anomaly_type,
                expected_T=expected_T,
            )
            if done is not None:
                resumed_cnt += 1
                if resumed_cnt % 500 == 0:
                    log(f"[BATCH {i}/{len(files)}] RESUME: {resumed_cnt} already-valid outputs skipped", log_fh=log_fh)
                continue

        log(
            f"[BATCH {i}/{len(files)}] INPUT: {fp} (anomaly_type={anomaly_type}, expected_T={expected_T})",
            log_fh=log_fh,
        )
        t0 = time.perf_counter()
        answer, payload, status, attempts = generate_json_validated(
            tokenizer=tokenizer,
            model=model,
            user_query=user_query,
            anomaly_type=anomaly_type,
            expected_T=expected_T,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
            coerce_after=coerce_after,
            log_fn=lambda s: log(s, log_fh=log_fh),
        )
        dt_s = time.perf_counter() - t0
        saved = save_output_json(answer, out_dir=out_dir, source_name=fp.stem, prefix="qwen_output")
        if status not in {"ok", "coerced"}:
            failed.append(f"{fp}\t{status}")
        log(f"[BATCH {i}/{len(files)}] STATUS={status} attempts={attempts} elapsed_s={dt_s:.3f}", log_fh=log_fh)
        log(f"[BATCH {i}/{len(files)}] OUTPUT:\n{answer}\n", log_fh=log_fh)
        log(f"[BATCH {i}/{len(files)}] SAVED: {saved}", log_fh=log_fh)

    _report_batch_result(out_dir=out_dir, resumed_cnt=resumed_cnt, failed=failed, log_fh=log_fh)
    return 0


def _report_batch_result(*, out_dir: str, resumed_cnt: int, failed: list, log_fh) -> None:
    """Log a batch summary.

    With the default max_retries=-1 nothing can land in `failed`; the list only
    fills if a finite --max-retries was requested explicitly.
    """
    if resumed_cnt:
        log(f"[BATCH] resumed (skipped already-valid): {resumed_cnt}", log_fh=log_fh)
    if failed:
        fail_path = Path(out_dir) / "failed_queries.tsv"
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        log(f"[BATCH] FAILED={len(failed)} -> {fail_path}", log_fh=log_fh)
    log("[BATCH DONE]", log_fh=log_fh)


def _batch_from_dir(
    *,
    batch_dir: str,
    out_dir: str,
    default_anomaly_type: Optional[str],
    max_new_tokens: int,
    max_retries: int,
    log_fh,
    resume: bool = True,
    coerce_after: int = DEFAULT_COERCE_AFTER,
) -> int:
    tokenizer, model = _get_model()
    files = iter_query_files(batch_dir)
    log(f"[BATCH] query_dir={batch_dir} (files={len(files)})", log_fh=log_fh)
    log(f"[BATCH] out_dir={out_dir} resume={resume}", log_fh=log_fh)

    resumed_cnt = 0
    failed: list[str] = []
    for i, fp in enumerate(files, start=1):
        raw = fp.read_text(encoding="utf-8", errors="replace")
        user_query = sanitize_user_query_text(raw)
        if not user_query:
            log(f"[BATCH {i}/{len(files)}] SKIP empty: {fp}", log_fh=log_fh)
            continue

        anomaly_type = infer_anomaly_type_from_filename(fp) or default_anomaly_type
        if anomaly_type is None:
            log(
                f"[BATCH {i}/{len(files)}] SKIP (cannot infer anomaly type; pass --default-anomaly-type): {fp.name}",
                log_fh=log_fh,
            )
            continue
        if anomaly_type == "A3":
            log(f"[BATCH {i}/{len(files)}] SKIP A3: {fp.name}", log_fh=log_fh)
            continue

        expected_T = infer_expected_T_from_path(fp)
        if resume:
            done = _find_valid_existing_output(
                out_dir=out_dir,
                source_name=fp.stem,
                anomaly_type=anomaly_type,
                expected_T=expected_T,
            )
            if done is not None:
                resumed_cnt += 1
                if resumed_cnt % 500 == 0:
                    log(f"[BATCH {i}/{len(files)}] RESUME: {resumed_cnt} already-valid outputs skipped", log_fh=log_fh)
                continue

        log(
            f"[BATCH {i}/{len(files)}] INPUT: {fp} (anomaly_type={anomaly_type}, expected_T={expected_T})",
            log_fh=log_fh,
        )
        t0 = time.perf_counter()
        answer, payload, status, attempts = generate_json_validated(
            tokenizer=tokenizer,
            model=model,
            user_query=user_query,
            anomaly_type=anomaly_type,
            expected_T=expected_T,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
            coerce_after=coerce_after,
            log_fn=lambda s: log(s, log_fh=log_fh),
        )
        dt_s = time.perf_counter() - t0
        saved = save_output_json(answer, out_dir=out_dir, source_name=fp.stem, prefix="qwen_output")
        if status not in {"ok", "coerced"}:
            failed.append(f"{fp}\t{status}")
        log(f"[BATCH {i}/{len(files)}] STATUS={status} attempts={attempts} elapsed_s={dt_s:.3f}", log_fh=log_fh)
        log(f"[BATCH {i}/{len(files)}] OUTPUT:\n{answer}\n", log_fh=log_fh)
        log(f"[BATCH {i}/{len(files)}] SAVED: {saved}", log_fh=log_fh)

    _report_batch_result(out_dir=out_dir, resumed_cnt=resumed_cnt, failed=failed, log_fh=log_fh)
    return 0


def _run_once(
    *,
    anomaly_type: Optional[str],
    out_dir: str,
    max_new_tokens: int,
    max_retries: int,
) -> int:
    if anomaly_type is None:
        print("[ERROR] --once requires --anomaly-type A1|A2", flush=True)
        return 2
    tokenizer, model = _get_model()
    user_query = sys.stdin.read().strip()
    if not user_query:
        return 0
    answer, payload, status, attempts = generate_json_validated(
        tokenizer=tokenizer,
        model=model,
        user_query=user_query,
        anomaly_type=anomaly_type,
        expected_T=None,
        max_new_tokens=max_new_tokens,
        max_retries=max_retries,
    )
    saved = save_output_json(answer, out_dir=out_dir, prefix="qwen_output")
    print(answer, flush=True)
    print(f"[SAVED] {saved}", flush=True)
    return 0


def _run_interactive(
    *,
    anomaly_type: Optional[str],
    out_dir: str,
    max_new_tokens: int,
    max_retries: int,
) -> int:
    if anomaly_type is None:
        print("[ERROR] interactive mode requires --anomaly-type A1|A2", flush=True)
        return 2
    tokenizer, model = _get_model()
    print("Model loaded. Paste input; submit an empty line to run; type ':q' to quit.", flush=True)
    print(f"[CONFIG] out_dir={out_dir}", flush=True)
    while True:
        user_query = read_multiline("\n[INPUT] Paste Trajectory/Task:")
        if user_query in {"", ":quit"}:
            break
        answer, payload, status, attempts = generate_json_validated(
            tokenizer=tokenizer,
            model=model,
            user_query=user_query,
            anomaly_type=anomaly_type,
            expected_T=None,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
        )
        print("\n[OUTPUT]\n" + answer + "\n", flush=True)
        saved = save_output_json(answer, out_dir=out_dir, prefix="qwen_output")
        print(f"[SAVED] {saved}", flush=True)
    return 0


def main() -> int:
    out_dir = get_arg_value("--out-dir", DEFAULT_OUT_DIR) or DEFAULT_OUT_DIR
    log_dir = get_arg_value("--log-dir", DEFAULT_LOG_DIR) or DEFAULT_LOG_DIR

    max_new_tokens_s = get_arg_value("--max-new-tokens", None)
    max_new_tokens = int(max_new_tokens_s) if (max_new_tokens_s and max_new_tokens_s.isdigit()) else DEFAULT_MAX_NEW_TOKENS

    max_retries_s = get_arg_value("--max-retries", None)
    # -1 means unlimited; the default is finite so a model that keeps miscounting
    # cannot stall the whole batch on a single query.
    max_retries = (
        int(max_retries_s) if (max_retries_s and max_retries_s.lstrip('-').isdigit()) else DEFAULT_MAX_RETRIES
    )

    coerce_after_s = get_arg_value("--coerce-after", None)
    coerce_after = int(coerce_after_s) if (coerce_after_s and coerce_after_s.isdigit()) else DEFAULT_COERCE_AFTER

    resume = "--no-resume" not in sys.argv

    cli_anomaly_type = _parse_anomaly_type_flag(get_arg_value("--anomaly-type", None))
    default_anomaly_type = _parse_anomaly_type_flag(get_arg_value("--default-anomaly-type", None))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"qwen_run_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"

    with log_path.open("w", encoding="utf-8") as log_fh:
        log(f"[LOG START] {log_path}", log_fh=log_fh)

        batch_dir = get_arg_value("--batch-dir", None)
        batch_root = get_arg_value("--batch-root", None)
        if batch_dir is not None and batch_root is not None:
            log("[ERROR] --batch-dir and --batch-root cannot be used together.", log_fh=log_fh)
            return 2

        if batch_root is not None:
            return _batch_from_root(
                batch_root=batch_root,
                out_dir=out_dir,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                log_fh=log_fh,
                resume=resume,
                coerce_after=coerce_after,
            )

        if batch_dir is not None:
            return _batch_from_dir(
                batch_dir=batch_dir,
                out_dir=out_dir,
                default_anomaly_type=default_anomaly_type,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                log_fh=log_fh,
                resume=resume,
                coerce_after=coerce_after,
            )

    if "--once" in sys.argv:
        return _run_once(
            anomaly_type=cli_anomaly_type,
            out_dir=out_dir,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
        )

    return _run_interactive(
        anomaly_type=cli_anomaly_type,
        out_dir=out_dir,
        max_new_tokens=max_new_tokens,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    raise SystemExit(main())

