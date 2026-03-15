
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from kap_copd.sustained_vowels import build_inventory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio_root', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()
    df = build_inventory(Path(args.audio_root))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
