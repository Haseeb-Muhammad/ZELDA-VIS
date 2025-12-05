#!/usr/bin/env python3
"""
Prepare predictions and ground-truths for Cell Tracking Challenge evaluation,
run the `SEGMeasure` tool, record the filename mapping, and clean up.

Usage example:
  python3 prepare_and_run_evaluation.py \
    --gt /path/to/gt_folder \
    --pred /path/to/pred_folder \
    --eval /path/to/eval/software_or_executable \
    --dataset my_dataset_name

- create a temporary evaluation folder with structure: <name>_01/01_GT/SEG and 01_RES/
- copy/rename files to the naming convention: man_segXXXX.tif and maskXXXX.tif
- save a JSON mapping of original -> new filenames in results/<dataset>/
- run SEGMeasure (auto-detected inside eval path) and capture its output
- save evaluation results in results/<dataset>/ directory within eval software path
- delete the temporary folder (unless --keep-temp is specified)ts output
- delete the temporary folder (unless --keep-temp is specified)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_segmeasure(eval_path: Path):
    # If user passed the executable directly
    if eval_path.is_file() and os_access_exec(eval_path):
        return str(eval_path)

    # If a directory was passed, look for SEGMeasure inside
    if eval_path.is_dir():
        candidate = eval_path / 'SEGMeasure'
        if candidate.exists() and os_access_exec(candidate):
            return str(candidate)
    return None


def os_access_exec(p: Path):
    try:
        return p.exists() and os.access(str(p), os.X_OK)
    except Exception:
        return False


def extract_last_number(stem: str):
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if m:
        return int(m.group(1))
    return None


def prepare_temp_structure(gt_dir: Path, pred_dir: Path, seg_exec: str, dataset_name: str,
                           seq='01', digits=4, keep_temp=False, output_dir: Path = None):
    # Create temp dir
    tmp = Path(tempfile.mkdtemp(prefix='ctc_eval_'))
    dataset_dir = tmp / f"{dataset_name}_{seq}"
    gt_seg_dir = dataset_dir / f"{seq}_GT" / "SEG"
    res_dir = dataset_dir / f"{seq}_RES"
    gt_seg_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff')])
    gt_files = {p.name: p for p in gt_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff')}

    mapping = []
    used_numbers = set()
    seq_counter = 0

    for p in pred_files:
        pred_stem = p.stem.replace('_masks', '')
        # Find GT candidate by exact stem first
        gt_candidate = None
        for ext in ('.tif', '.tiff'):
            cand = gt_dir / f"{pred_stem}{ext}"
            if cand.exists():
                gt_candidate = cand
                break

        # If not found, try substring match
        if gt_candidate is None:
            for g in gt_dir.iterdir():
                if g.is_file() and pred_stem in g.stem:
                    gt_candidate = g
                    break

        # If still not found, try to match numeric frame
        frame = extract_last_number(pred_stem)
        if gt_candidate is None and frame is not None:
            # look for gt file containing same number (with or without leading zeros)
            frame_str = str(frame)
            for g in gt_dir.iterdir():
                if g.is_file() and frame_str in g.stem:
                    gt_candidate = g
                    break

        # Ensure unique numeric id for output file
        if frame is None:
            # fallback to sequential counter
            while seq_counter in used_numbers:
                seq_counter += 1
            frame = seq_counter
            seq_counter += 1

        # If number collides, find next free
        if frame in used_numbers:
            # increment until free
            original = frame
            while frame in used_numbers:
                frame += 1
        used_numbers.add(frame)

        mask_name = f"mask{frame:0{digits}d}.tif"
        seg_name = f"man_seg{frame:0{digits}d}.tif"

        dest_pred = res_dir / mask_name
        shutil.copy2(p, dest_pred)

        if gt_candidate is not None:
            dest_gt = gt_seg_dir / seg_name
            shutil.copy2(gt_candidate, dest_gt)
            gt_path_str = str(gt_candidate)
        else:
            dest_gt = None
            gt_path_str = None

        mapping.append({
            'original_pred': str(p),
            'original_gt': gt_path_str,
            'new_pred': str(dest_pred),
            'new_gt': str(dest_gt) if dest_gt else None,
            'frame': frame,
        })

    # write mapping to temp first
    mapping_file_temp = tmp / 'mapping.json'
    with mapping_file_temp.open('w') as f:
        json.dump(mapping, f, indent=2)

    # Run evaluation
    eval_log_temp = tmp / 'eval_log.txt'
    # SEGMeasure usage: SEGMeasure <dir> <seq> <num_digits>
    cmd = [seg_exec, str(dataset_dir), seq, str(digits)]
    print('Running:', ' '.join(cmd))
    with eval_log_temp.open('w') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    print(f"SEGMeasure exit code: {proc.returncode}")
    
    # Read and display evaluation results
    if eval_log_temp.exists():
        print("\n=== Evaluation Results ===")
        with eval_log_temp.open('r') as f:
            log_content = f.read()
            print(log_content)
        print("=" * 30)

    # Copy results to output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        mapping_file = output_dir / 'mapping.json'
        eval_log = output_dir / 'evaluation_results.txt'
        
        shutil.copy2(mapping_file_temp, mapping_file)
        shutil.copy2(eval_log_temp, eval_log)
        
        print(f"Results saved to: {output_dir}")
        print(f"  - Mapping: {mapping_file}")
        print(f"  - Evaluation log: {eval_log}")
    else:
        mapping_file = mapping_file_temp
        eval_log = eval_log_temp

    # Optionally keep or cleanup
    if keep_temp:
        print(f"Temporary evaluation folder preserved at: {tmp}")
        return proc.returncode, tmp, mapping_file, eval_log
    else:
        # remove the temp dir
        try:
            shutil.rmtree(tmp)
        except Exception as e:
            print(f"Warning: failed to remove temp dir {tmp}: {e}")
        return proc.returncode, None, mapping_file, eval_log


def main():
    parser = argparse.ArgumentParser(description='Prepare and run CTC SEGMeasure on GT and predictions')
    parser.add_argument('--gt', required=True, help='Path to ground truth folder')
    parser.add_argument('--pred', required=True, help='Path to predictions folder')
    parser.add_argument('--eval', required=True, help='Path to evaluation software directory or SEGMeasure executable')
    parser.add_argument('--dataset', required=True, help='Dataset name for organizing results')
    parser.add_argument('--name', default='TMP', help='Base dataset name to create (default: TMP)')
    parser.add_argument('--seq', default='01', help='Sequence id to use (default: 01)')
    parser.add_argument('--digits', type=int, default=4, help='Zero-padding digits for output filenames (default: 4)')
    parser.add_argument('--keep-temp', action='store_true', help='Do not delete temporary folder after run')
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    eval_path = Path(args.eval)

    if not gt_dir.exists() or not gt_dir.is_dir():
        print('GT path not found or not a directory:', gt_dir)
        sys.exit(2)
    if not pred_dir.exists() or not pred_dir.is_dir():
        print('Predictions path not found or not a directory:', pred_dir)
        sys.exit(2)

    seg_exec = find_segmeasure(eval_path)
    if seg_exec is None:
        print('SEGMeasure executable not found inside provided eval path:', eval_path)
        sys.exit(3)

    # Determine output directory
    eval_dir = eval_path if eval_path.is_dir() else eval_path.parent
    results_dir = eval_dir / 'results' / args.dataset
    
    code, tmpdir, mapping_file, log_file = prepare_temp_structure(
        gt_dir, pred_dir, seg_exec, dataset_name=args.name, seq=args.seq, 
        digits=args.digits, keep_temp=args.keep_temp, output_dir=results_dir
    )

    if code != 0:
        print('Evaluation returned non-zero exit code:', code)

    # Print locations (mapping/log) if preserved
    if tmpdir:
        print('Temporary dir:', tmpdir)
    print('Mapping file (or path that was removed):', mapping_file)
    print('Evaluation log (or path that was removed):', log_file)
    sys.exit(code)


if __name__ == '__main__':
    main()
