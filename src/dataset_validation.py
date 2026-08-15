import os
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import statistics
from PIL import Image

from src.utils import load_config, save_config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 hash of file content for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Calculate difference hash (dhash) of PIL Image for near-duplicate analysis.
    Uses pure PIL without external heavy dependencies.
    """
    # Resize image to (hash_size + 1, hash_size) in grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    if hasattr(resized, "get_flattened_data"):
        pixels = list(resized.get_flattened_data())
    else:
        pixels = list(resized.getdata())
    
    # Compare adjacent pixels in each row
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
            
    # Convert binary array to hex string
    decimal_value = 0
    hex_str = []
    for index, val in enumerate(difference):
        if val:
            decimal_value += 2 ** (index % 4)
        if (index % 4) == 3:
            hex_str.append(hex(decimal_value)[2:])
            decimal_value = 0
    return "".join(hex_str)


def validate_single_image(filepath: Path, supported_exts: Set[str] = SUPPORTED_EXTENSIONS) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """
    Validate that an image exists, has supported extension, can be opened/verified/loaded,
    and has valid non-zero dimensions.
    Returns: (is_valid, reason, (width, height))
    """
    if not filepath.is_file():
        return False, "File does not exist", None
    
    ext = filepath.suffix.lower()
    if ext not in supported_exts:
        return False, f"Unsupported extension: {ext}", None

    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid dimensions: {w}x{h}", None
            return True, "Valid", (w, h)
    except Exception as e:
        return False, f"Corrupted image ({type(e).__name__}: {str(e)})", None


def check_split_integrity(split_name: str, split_dir: str, supported_exts: Set[str] = SUPPORTED_EXTENSIONS) -> Dict[str, Any]:
    """
    Inspect a dataset split directory, validate files, and collect size statistics.
    """
    path = Path(split_dir)
    report = {
        "split": split_name,
        "split_path": str(path),
        "exists": path.exists(),
        "classes": [],
        "num_classes": 0,
        "class_counts": {},
        "empty_classes": [],
        "total_files": 0,
        "valid_images": 0,
        "corrupted_files": [],
        "unsupported_files": [],
        "invalid_dimension_files": [],
        "image_sizes": [] # List of (w, h)
    }

    if not path.exists():
        return report

    class_dirs = sorted([d for d in path.iterdir() if d.is_dir()], key=lambda d: d.name)
    report["num_classes"] = len(class_dirs)
    report["classes"] = [d.name for d in class_dirs]

    for cdir in class_dirs:
        class_name = cdir.name
        count = 0
        file_list = []
        for root, _, files in os.walk(cdir):
            for file in files:
                file_list.append(Path(root) / file)

        if len(file_list) == 0:
            report["empty_classes"].append(class_name)

        for file_path in file_list:
            report["total_files"] += 1
            if file_path.suffix.lower() not in supported_exts:
                report["unsupported_files"].append({
                    "split": split_name,
                    "class": class_name,
                    "path": str(file_path),
                    "problem": f"Unsupported extension {file_path.suffix}"
                })
                continue

            is_valid, reason, dims = validate_single_image(file_path, supported_exts)
            if not is_valid:
                prob_dict = {
                    "split": split_name,
                    "class": class_name,
                    "path": str(file_path),
                    "problem": reason
                }
                if "Invalid dimensions" in reason:
                    report["invalid_dimension_files"].append(prob_dict)
                else:
                    report["corrupted_files"].append(prob_dict)
            else:
                count += 1
                report["valid_images"] += 1
                if dims:
                    report["image_sizes"].append(dims)
        
        report["class_counts"][class_name] = count

    return report


def detect_duplicates_and_leakage(split_paths: Dict[str, str], supported_exts: Set[str] = SUPPORTED_EXTENSIONS) -> Dict[str, Any]:
    """
    Detect exact SHA-256 duplicates across train, val, and test splits.
    Distinguishes intra-split duplicates from cross-split data leakage.
    Also computes perceptual dhash for near-duplicate detection.
    """
    exact_hashes: Dict[str, List[Dict[str, str]]] = {}
    dhashes: Dict[str, List[Dict[str, str]]] = {}

    for split_name, path_str in split_paths.items():
        path = Path(path_str)
        if not path.exists():
            continue

        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_exts:
                    try:
                        sha_val = compute_sha256(file_path)
                        class_name = file_path.parent.name
                        entry = {
                            "split": split_name,
                            "class": class_name,
                            "path": str(file_path),
                            "hash": sha_val
                        }
                        if sha_val not in exact_hashes:
                            exact_hashes[sha_val] = []
                        exact_hashes[sha_val].append(entry)

                        # Near duplicate analysis using dhash
                        with Image.open(file_path) as img:
                            dh = compute_dhash(img)
                            if dh not in dhashes:
                                dhashes[dh] = []
                            dhashes[dh].append({
                                "split": split_name,
                                "class": class_name,
                                "path": str(file_path),
                                "dhash": dh
                            })
                    except Exception:
                        pass

    # Exact Duplicates Breakdown
    intra_split_duplicates = []
    cross_split_leakage = []

    for sha_val, locations in exact_hashes.items():
        if len(locations) > 1:
            splits_present = set(loc["split"] for loc in locations)
            group_info = {
                "hash": sha_val,
                "count": len(locations),
                "splits": sorted(list(splits_present)),
                "locations": locations
            }
            if len(splits_present) > 1:
                cross_split_leakage.append(group_info)
            else:
                intra_split_duplicates.append(group_info)

    # Near Duplicates Breakdown (dhash matches with count > 1 that are not exact sha matches)
    near_duplicates = []
    for dh_val, locations in dhashes.items():
        if len(locations) > 1:
            # Check if all files in this group have the same SHA hash (if so, they are exact duplicates, not just near)
            sha_set = set(compute_sha256(Path(loc["path"])) for loc in locations if Path(loc["path"]).exists())
            if len(sha_set) > 1:
                near_duplicates.append({
                    "dhash": dh_val,
                    "count": len(locations),
                    "locations": locations
                })

    return {
        "total_unique_exact_hashes": len(exact_hashes),
        "intra_split_duplicates": intra_split_duplicates,
        "intra_split_duplicate_count": len(intra_split_duplicates),
        "cross_split_leakage": cross_split_leakage,
        "cross_split_leakage_count": len(cross_split_leakage),
        "has_leakage": len(cross_split_leakage) > 0,
        "near_duplicate_pairs": near_duplicates,
        "near_duplicate_count": len(near_duplicates)
    }


def validate_class_alignment(train_dir: str, val_dir: str, test_dir: str) -> Dict[str, Any]:
    """Inspect and validate class folder alignment across train, val, and test splits."""
    def get_classes(p_str):
        p = Path(p_str)
        if not p.exists():
            return set()
        return set(d.name for d in p.iterdir() if d.is_dir())

    train_cls = get_classes(train_dir)
    val_cls = get_classes(val_dir)
    test_cls = get_classes(test_dir)

    all_cls = train_cls | val_cls | test_cls
    aligned = (train_cls == val_cls == test_cls) if (train_cls and val_cls and test_cls) else False

    missing_in_val = sorted(list(train_cls - val_cls))
    missing_in_test = sorted(list(train_cls - test_cls))
    extra_in_val = sorted(list(val_cls - train_cls))
    extra_in_test = sorted(list(test_cls - train_cls))

    return {
        "aligned": aligned,
        "train_classes": sorted(list(train_cls)),
        "val_classes": sorted(list(val_cls)),
        "test_classes": sorted(list(test_cls)),
        "total_unique_classes": len(all_cls),
        "train_class_count": len(train_cls),
        "val_class_count": len(val_cls),
        "test_class_count": len(test_cls),
        "missing_in_val": missing_in_val,
        "missing_in_test": missing_in_test,
        "extra_in_val": extra_in_val,
        "extra_in_test": extra_in_test
    }


def calculate_distribution_stats(class_counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculate min, max, mean, median class sizes and imbalance ratio."""
    counts = list(class_counts.values())
    if not counts:
        return {
            "total_classes": 0,
            "total_images": 0,
            "min_class_size": 0,
            "max_class_size": 0,
            "mean_class_size": 0.0,
            "median_class_size": 0.0,
            "imbalance_ratio": 1.0
        }
    
    min_c = min(counts)
    max_c = max(counts)
    mean_c = round(statistics.mean(counts), 2)
    median_c = round(statistics.median(counts), 2)
    imbalance = round(max_c / min_c, 2) if min_c > 0 else float("inf")

    return {
        "total_classes": len(counts),
        "total_images": sum(counts),
        "min_class_size": min_c,
        "max_class_size": max_c,
        "mean_class_size": mean_c,
        "median_class_size": median_c,
        "imbalance_ratio": imbalance
    }


def export_class_distribution_csv(
    split_reports: Dict[str, Dict[str, Any]],
    output_path: str = "results/dataset_class_distribution.csv"
) -> None:
    """Save class distribution CSV report containing split, class_name, image_count, percentage."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name, report in split_reports.items():
        total_images = report.get("valid_images", 0)
        class_counts = report.get("class_counts", {})
        for class_name, count in sorted(class_counts.items()):
            percentage = round((count / total_images * 100), 4) if total_images > 0 else 0.0
            rows.append({
                "split": split_name,
                "class_name": class_name,
                "image_count": count,
                "percentage": percentage
            })

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "image_count", "percentage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Validation] Class distribution CSV saved to: {out_p}")


def infer_and_save_class_mapping(train_dir: str, output_path: str = "results/class_to_idx.json") -> Dict[str, int]:
    """Infer deterministic class_to_idx mapping from train directory and save as JSON."""
    p = Path(train_dir)
    if not p.exists():
        raise FileNotFoundError(f"Train directory '{train_dir}' does not exist.")

    class_dirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=4)

    print(f"[Validation] Inferred {len(class_to_idx)} classes from train directory. Saved to: {out_p}")
    return class_to_idx


def compute_image_size_stats(split_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall min/max/average width and height across all splits."""
    all_sizes = []
    for report in split_reports.values():
        all_sizes.extend(report.get("image_sizes", []))

    if not all_sizes:
        return {"min_width": 0, "max_width": 0, "avg_width": 0.0, "min_height": 0, "max_height": 0, "avg_height": 0.0}

    widths = [s[0] for s in all_sizes]
    heights = [s[1] for s in all_sizes]

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": round(statistics.mean(widths), 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": round(statistics.mean(heights), 2),
        "total_scanned_images": len(all_sizes)
    }


def run_full_dataset_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute comprehensive dataset inspection, file integrity checks, duplicate scan,
    class distribution analysis, dynamic class mapping, and report generation.
    """
    data_cfg = config.get("data", {})
    train_dir = data_cfg.get("train_dir", "data/train")
    val_dir = data_cfg.get("val_dir", "data/val")
    test_dir = data_cfg.get("test_dir", "data/test")

    print("==================================================")
    print("PHASE 1 & 2: DATASET DIRECTORY & INTEGRITY INSPECTION")
    print("==================================================")

    train_report = check_split_integrity("train", train_dir)
    val_report = check_split_integrity("val", val_dir)
    test_report = check_split_integrity("test", test_dir)

    split_reports = {"train": train_report, "val": val_report, "test": test_report}

    print(f"Train split: {train_report['num_classes']} classes, {train_report['valid_images']} valid images")
    print(f"Val split:   {val_report['num_classes']} classes, {val_report['valid_images']} valid images")
    print(f"Test split:  {test_report['num_classes']} classes, {test_report['valid_images']} valid images")

    print("\n==================================================")
    print("PHASE 2: CLASS ALIGNMENT SCAN")
    print("==================================================")
    alignment_report = validate_class_alignment(train_dir, val_dir, test_dir)
    if not alignment_report["aligned"]:
        print("[WARNING] Class set mismatch across splits!")
        if alignment_report["missing_in_val"]:
            print(f"  Missing in Validation: {alignment_report['missing_in_val']}")
        if alignment_report["missing_in_test"]:
            print(f"  Missing in Test:       {alignment_report['missing_in_test']}")
    else:
        print("[PASS] Classes match perfectly across train, val, and test.")

    print("\n==================================================")
    print("PHASE 3 & 4 & 5: FILE VALIDATION & DUPLICATE / LEAKAGE SCAN")
    print("==================================================")
    dup_report = detect_duplicates_and_leakage({
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    })

    all_corrupted = (
        train_report["corrupted_files"] +
        val_report["corrupted_files"] +
        test_report["corrupted_files"]
    )
    all_unsupported = (
        train_report["unsupported_files"] +
        val_report["unsupported_files"] +
        test_report["unsupported_files"]
    )

    print(f"Corrupted Images:       {len(all_corrupted)}")
    print(f"Unsupported Files:      {len(all_unsupported)}")
    print(f"Intra-Split Duplicates: {dup_report['intra_split_duplicate_count']}")
    print(f"Cross-Split Leakage:    {dup_report['cross_split_leakage_count']}")
    print(f"Near-Duplicate Pairs:   {dup_report['near_duplicate_count']}")

    print("\n==================================================")
    print("PHASE 6 & 7: CLASS DISTRIBUTION & DYNAMIC MAPPING")
    print("==================================================")
    export_class_distribution_csv(split_reports)
    class_to_idx = infer_and_save_class_mapping(train_dir)

    dist_stats = {
        "train": calculate_distribution_stats(train_report["class_counts"]),
        "val": calculate_distribution_stats(val_report["class_counts"]),
        "test": calculate_distribution_stats(test_report["class_counts"])
    }

    print("\n==================================================")
    print("PHASE 8: IMAGE SIZE ANALYSIS")
    print("==================================================")
    size_stats = compute_image_size_stats(split_reports)
    print(f"Widths:  Min={size_stats['min_width']}, Max={size_stats['max_width']}, Avg={size_stats['avg_width']}")
    print(f"Heights: Min={size_stats['min_height']}, Max={size_stats['max_height']}, Avg={size_stats['avg_height']}")

    # Determine overall status
    if len(all_corrupted) > 0 or dup_report["has_leakage"]:
        status = "FAIL"
    elif not alignment_report["aligned"] or dup_report["intra_split_duplicate_count"] > 0 or dup_report["near_duplicate_count"] > 0:
        status = "WARNING"
    else:
        status = "PASS"

    full_report = {
        "status": status,
        "dataset_paths": {
            "train": train_dir,
            "val": val_dir,
            "test": test_dir
        },
        "split_sizes": {
            "train": train_report["valid_images"],
            "val": val_report["valid_images"],
            "test": test_report["valid_images"],
            "total": train_report["valid_images"] + val_report["valid_images"] + test_report["valid_images"]
        },
        "class_count": len(class_to_idx),
        "class_names": sorted(list(class_to_idx.keys())),
        "class_mapping": class_to_idx,
        "class_alignment": alignment_report,
        "class_distribution_stats": dist_stats,
        "corrupted_images": all_corrupted,
        "unsupported_files": all_unsupported,
        "empty_classes": {
            "train": train_report["empty_classes"],
            "val": val_report["empty_classes"],
            "test": test_report["empty_classes"]
        },
        "duplicate_detection": {
            "intra_split_duplicates": dup_report["intra_split_duplicates"],
            "intra_split_duplicate_count": dup_report["intra_split_duplicate_count"],
            "cross_split_leakage": dup_report["cross_split_leakage"],
            "cross_split_leakage_count": dup_report["cross_split_leakage_count"],
            "near_duplicates": dup_report["near_duplicate_pairs"],
            "near_duplicate_count": dup_report["near_duplicate_count"]
        },
        "image_size_statistics": size_stats
    }

    # Clean reports dictionary for json serializing (omit raw image sizes array)
    json_report = full_report.copy()
    result_dir = Path(config.get("paths", {}).get("result_dir", "results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    report_file = result_dir / "dataset_validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=4)

    print(f"\n[Validation] Validation report written to: {report_file}")
    print(f"[Validation Status] {status}")

    return full_report


def main():
    """CLI entry point for python -m src.dataset_validation"""
    config = load_config()
    report = run_full_dataset_validation(config)
    
    print("\n==================================================")
    print("MILESTONE 2 — DATASET VALIDATION SUMMARY")
    print("==================================================")
    print(f"Overall Status:            {report['status']}")
    print(f"Total Unique Classes:      {report['class_count']}")
    print(f"Train Images:              {report['split_sizes']['train']}")
    print(f"Validation Images:         {report['split_sizes']['val']}")
    print(f"Test Images:               {report['split_sizes']['test']}")
    print(f"Corrupted Images:          {len(report['corrupted_images'])}")
    print(f"Unsupported Files:         {len(report['unsupported_files'])}")
    print(f"Exact Intra-Split Dups:    {report['duplicate_detection']['intra_split_duplicate_count']}")
    print(f"Cross-Split Leakage:       {report['duplicate_detection']['cross_split_leakage_count']}")
    print("==================================================")

    if report["status"] == "FAIL":
        print("\n[FAIL] Serious corrupted files or data leakage detected.")
        exit(1)


if __name__ == "__main__":
    main()
