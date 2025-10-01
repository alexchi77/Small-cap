import pandas as pd
from pathlib import Path
from src.data_ingest import load_local_halts, enrich_halts_with_bars_and_news, cfg  # <-- your module filename

def main():
    # 1. Load halts data
    halts_df = load_local_halts(cfg["data_path"])  # must point to your halts CSV
    
    # 2. Enrich halts with 5-min bars and news over +/- 7 days
    enriched = enrich_halts_with_bars_and_news(
        halts_df,
        pre_days=1,
        post_days=7,
        bar_timespan="minute"
    )

    # 3. Save results (parquet recommended since we have nested dicts)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Save as pickle (keeps nested objects)
    out_path = out_dir / "halts_enriched.pkl"
    enriched.to_pickle(out_path)
    print(f"Enriched halts saved to {out_path}")

    # Optionally also save a flat version (without bars/news) for preview
    meta_cols = ["ticker", "halt_time", "resume_time", "reason"]
    enriched[meta_cols].to_csv(out_dir / "halts_meta.csv", index=False)
    print(f"Metadata saved to {out_dir / 'halts_meta.csv'}")


if __name__ == "__main__":
    main()
