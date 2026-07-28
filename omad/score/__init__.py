"""Score: LLM Anomaly Scoring"""


def run_score(
    T: int | None = None,
    batch_root: str | None = None,
    batch_dir: str | None = None,
    anomaly_type: str | None = None,
    max_new_tokens: int = 1024,
    max_retries: int = 6,
    output_dir: str | None = None,
    log_dir: str | None = None,
    interactive: bool = False,
    stdin_mode: bool = False,
    verbose: bool = False,
    resume: bool = True,
):
    """
    Score: Generate LLM anomaly suitability scores.

    Args:
        T: Time slice (12/24) - auto-derives paths
        batch_root: Batch root with user_query/ subdirs
        batch_dir: Flat directory with .txt files
        anomaly_type: Force anomaly type (A1/A2)
        max_new_tokens: Max LLM tokens
        max_retries: Validation retries (-1=unlimited)
        output_dir: JSON output directory
        log_dir: Log directory
        interactive: Interactive prompt mode
        stdin_mode: Single query from stdin
        verbose: Detailed logging
        resume: Skip queries that already have a valid saved output
    """
    from datetime import datetime
    from pathlib import Path
    from omad.config import get_config
    from omad.paths import resolve_stage2_paths
    from omad.utils.logging import console, log_stage_header, log_config, log_success
    import typer

    log_stage_header(2, "LLM Anomaly Scoring")

    config = get_config()

    # Resolve paths
    if T is not None and not batch_root:
        paths = resolve_stage2_paths(T, config, output_dir=output_dir, log_dir=log_dir)
        batch_root = str(paths["batch_root"])
        output_dir = str(paths["output_dir"])
        log_dir = str(paths["log_dir"])

    # Display configuration
    log_config({
        "Batch Root": batch_root or "N/A",
        "Batch Dir": batch_dir or "N/A",
        "Anomaly Type": anomaly_type or "Auto-detect",
        "Max Tokens": max_new_tokens,
        "Max Retries": max_retries if max_retries >= 0 else "Unlimited",
        "Resume": resume,
        "Output Dir": output_dir,
        "Log Dir": log_dir,
    })

    # Import score functions lazily (avoids loading model at import time)
    from omad.score.main import _batch_from_root, _batch_from_dir, _run_once, _run_interactive

    # Execute based on mode
    try:
        if interactive:
            _run_interactive(
                anomaly_type=anomaly_type,
                out_dir=output_dir or "outputs",
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
            )
        elif stdin_mode:
            _run_once(
                anomaly_type=anomaly_type,
                out_dir=output_dir or "outputs",
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
            )
        elif batch_root:
            # Create log file for batch processing
            log_path = Path(log_dir or "logs")
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"qwen_run_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"

            with log_file.open("w", encoding="utf-8") as log_fh:
                _batch_from_root(
                    batch_root=batch_root,
                    out_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                    max_retries=max_retries,
                    log_fh=log_fh,
                    resume=resume,
                )
            console.print(f"  Log saved to: {log_file}")
        elif batch_dir:
            log_path = Path(log_dir or "logs")
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"qwen_run_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"

            with log_file.open("w", encoding="utf-8") as log_fh:
                _batch_from_dir(
                    batch_dir=batch_dir,
                    out_dir=output_dir,
                    default_anomaly_type=anomaly_type,
                    max_new_tokens=max_new_tokens,
                    max_retries=max_retries,
                    log_fh=log_fh,
                    resume=resume,
                )
            console.print(f"  Log saved to: {log_file}")
        else:
            console.print("[red]Error:[/red] Must specify --T, --batch-root, or --batch-dir")
            raise typer.Exit(1)

        log_success("Score completed successfully\n")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

# Alias for backward compatibility
run_stage2 = run_score
